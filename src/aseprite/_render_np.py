"""NumPy compositing: :mod:`aseprite._render`'s per-pixel blenders over whole
arrays - the same integer arithmetic (``MUL_UN8``, truncating division) and
the same floating point for the HSL modes, so the bytes are the same as the
pure-Python path's (tests compare them, mode by mode). Used when NumPy is
installed; ``ASEPRITE_PURE_PYTHON=1`` turns it off.

Arrays here are ``int32`` ``(N, 4)`` RGBA rows - wide enough for every
intermediate (at most ``255 * 255 * 2``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aseprite import _render as _r
from aseprite._model import BlendMode, Cel, ColorMode, Layer, Pixels

if TYPE_CHECKING:
    from aseprite._sprite import Sprite


def mul_un8(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    """``a * b / 255`` rounded as Aseprite's ``MUL_UN8`` (arithmetic shifts:
    a negative ``a`` rounds as the scalar version's Python ints do)."""
    t = a * b + 0x80
    return ((t >> 8) + t) >> 8


def _div_trunc(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """C integer division: truncates toward zero (``den`` > 0)."""
    q = np.abs(num) // den
    return np.where(num < 0, -q, q)


def blend_normal(d: np.ndarray, s: np.ndarray, opacity: int) -> np.ndarray:
    """:func:`aseprite._render._blend_normal` over rows."""
    out = d.copy()
    da, s_raw = d[:, 3], s[:, 3]
    sa = mul_un8(s_raw, opacity)
    empty = da == 0
    if empty.any():
        out[empty, :3] = s[empty, :3]
        out[empty, 3] = sa[empty]
    mix = ~empty & (s_raw != 0)
    if mix.any():
        dm, sm, sam = d[mix], s[mix], sa[mix]
        dam = dm[:, 3]
        out_a = sam + dam - mul_un8(dam, sam)            # > 0: dam > 0
        delta = (sm[:, :3] - dm[:, :3]) * sam[:, None]
        out[mix, :3] = dm[:, :3] + _div_trunc(delta, out_a[:, None])
        out[mix, 3] = out_a
    return out


def blend_merge(d: np.ndarray, s: np.ndarray, opacity: np.ndarray | int) -> np.ndarray:
    """:func:`aseprite._render._blend_merge` - ``opacity`` one value or one a row."""
    op = opacity[:, None] if isinstance(opacity, np.ndarray) else opacity
    moved = d[:, :3] + mul_un8(s[:, :3] - d[:, :3], op)
    rgb = np.where(d[:, 3:4] == 0, s[:, :3], np.where(s[:, 3:4] == 0, d[:, :3], moved))
    alpha = d[:, 3] + mul_un8(s[:, 3] - d[:, 3], opacity)
    rgb = np.where(alpha[:, None] == 0, 0, rgb)
    return np.concatenate([rgb, alpha[:, None]], axis=1)


# -- the modes' colors --------------------------------------------------------

_LUTS: dict[BlendMode, np.ndarray] = {}


def _lut(mode: BlendMode) -> np.ndarray:
    """A separable mode as a table ``[backdrop, source] -> byte`` - made with
    the pure-Python function itself, so it can't differ from it."""
    table = _LUTS.get(mode)
    if table is None:
        fn = _r._SEPARABLE[mode]
        table = np.array([[fn(b, s) for s in range(256)] for b in range(256)],
                         dtype=np.int32)
        _LUTS[mode] = table
    return table


def _lum(r, g, b):
    return 0.3 * r + 0.59 * g + 0.11 * b


def _clip_color(r, g, b):
    lum = _lum(r, g, b)
    n = np.minimum(np.minimum(r, g), b)
    x = np.maximum(np.maximum(r, g), b)
    low = n < 0
    if low.any():
        den = np.where(low, lum - n, 1.0)
        r = np.where(low, lum + (((r - lum) * lum) / den), r)
        g = np.where(low, lum + (((g - lum) * lum) / den), g)
        b = np.where(low, lum + (((b - lum) * lum) / den), b)
    high = x > 1
    if high.any():
        den = np.where(high, x - lum, 1.0)
        r = np.where(high, lum + (((r - lum) * (1 - lum)) / den), r)
        g = np.where(high, lum + (((g - lum) * (1 - lum)) / den), g)
        b = np.where(high, lum + (((b - lum) * (1 - lum)) / den), b)
    return r, g, b


def _set_lum(r, g, b, lum):
    d = lum - _lum(r, g, b)
    return _clip_color(r + d, g + d, b + d)


def _sat(r, g, b):
    return np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)


def _set_sat(r, g, b, sat):
    lo = np.minimum(np.minimum(r, g), b)
    span = np.maximum(np.maximum(r, g), b) - lo
    ok = span > 0.0
    den = np.where(ok, span, 1.0)
    return (np.where(ok, ((r - lo) * sat) / den, 0.0),
            np.where(ok, ((g - lo) * sat) / den, 0.0),
            np.where(ok, ((b - lo) * sat) / den, 0.0))


def _unit(c: np.ndarray):
    return c[:, 0] / 255.0, c[:, 1] / 255.0, c[:, 2] / 255.0


def _hsl(mode: BlendMode, d: np.ndarray, s: np.ndarray):
    br, bg, bb = _unit(d)
    if mode is BlendMode.HUE:
        r, g, b = _set_sat(*_unit(s), _sat(br, bg, bb))
        return _set_lum(r, g, b, _lum(br, bg, bb))
    if mode is BlendMode.SATURATION:
        r, g, b = _set_sat(br, bg, bb, _sat(*_unit(s)))
        return _set_lum(r, g, b, _lum(br, bg, bb))
    if mode is BlendMode.COLOR:
        return _set_lum(*_unit(s), _lum(br, bg, bb))
    return _set_lum(br, bg, bb, _lum(*_unit(s)))           # LUMINOSITY


def _classic(mode: BlendMode, d: np.ndarray, s: np.ndarray, opacity: int) -> np.ndarray:
    """The mode's color, then composited as Normal (``rgba_blender_<mode>``)."""
    if mode in _r._SEPARABLE:
        lut = _lut(mode)
        color = lut[d[:, :3], s[:, :3]]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rgb = _hsl(mode, d, s)
        # a C int cast of 255 * value (truncation), kept in a byte
        color = np.stack([np.clip(np.trunc(255.0 * v), 0, 255) for v in rgb],
                         axis=1).astype(np.int32)
    return blend_normal(d, np.concatenate([color, s[:, 3:4]], axis=1), opacity)


def blend(mode: BlendMode, new_blend: bool, grayscale: bool,
          d: np.ndarray, s: np.ndarray, opacity: int) -> np.ndarray:
    """:func:`aseprite._render.get_blender`'s blender, over rows."""
    if grayscale:
        if mode in _r._NON_SEPARABLE:
            return blend_normal(d, s, opacity)
        if mode is BlendMode.ADDITION and new_blend:
            mode = BlendMode.EXCLUSION
    if mode is BlendMode.NORMAL or (mode not in _r._SEPARABLE
                                    and mode not in _r._NON_SEPARABLE):
        return blend_normal(d, s, opacity)
    if not new_blend:
        return _classic(mode, d, s, opacity)
    # RGBA_BLENDER_N: the mode faded toward Normal by the backdrop's alpha
    normal = blend_normal(d, s, opacity)
    color = _classic(mode, d, s, opacity)
    da = d[:, 3]
    merged = blend_merge(normal, color, da)
    composite = mul_un8(da, mul_un8(s[:, 3], opacity))
    out = blend_merge(merged, color, composite)
    return np.where((da == 0)[:, None], normal, out)


# -- frames -------------------------------------------------------------------

def rgba(sprite: Sprite, layer: Layer, pixels: Pixels) -> np.ndarray:
    """A cel's pixels as ``(h, w, 4)`` RGBA bytes (:func:`aseprite._render._pixel_rgba`)."""
    data = np.frombuffer(pixels.data, dtype=np.uint8)
    h, w = pixels.height, pixels.width
    if pixels.color_mode is ColorMode.RGBA:
        return data[: h * w * 4].reshape(h, w, 4)
    if pixels.color_mode is ColorMode.GRAYSCALE:
        ga = data[: h * w * 2].reshape(h, w, 2)
        return np.stack([ga[..., 0], ga[..., 0], ga[..., 0], ga[..., 1]], axis=-1)
    palette = sprite.palette
    table = _palette_table([palette[i] for i in range(min(len(palette), 256))])
    if not layer.background and 0 <= sprite.transparent_index < 256:
        table[sprite.transparent_index] = 0
    return table[data[: h * w].reshape(h, w)]


def _palette_table(colors) -> np.ndarray:
    """``(256, 4)`` RGBA bytes: the colors, the rest zero."""
    table = np.zeros((256, 4), dtype=np.uint8)
    if colors:
        table[: len(colors)] = np.array([(c.r, c.g, c.b, c.a) for c in colors],
                                        dtype=np.uint8)
    return table


def apply_palette(palette, indices: bytearray, painted: bytearray) -> bytes:
    """:func:`aseprite._render._apply_palette`: indices to RGBA - unpainted
    pixels zero."""
    table = _palette_table([palette[i] for i in range(min(len(palette), 256))]
                           if not hasattr(palette, "colors") else palette.colors[:256])
    # one 32-bit gather a pixel (4x faster than gathering the bytes)
    out = np.take(table.view(np.uint32).ravel(), np.frombuffer(indices, dtype=np.uint8))
    out[np.frombuffer(painted, dtype=np.uint8) == 0] = 0
    return out.tobytes()


def _canvas(dest: bytearray, sprite: Sprite) -> np.ndarray:
    return np.frombuffer(dest, dtype=np.uint8).reshape(sprite.height, sprite.width, 4)


def _blend_region(region: np.ndarray, src: np.ndarray, mode: BlendMode, new_blend: bool,
                  grayscale: bool, opacity: int) -> None:
    """Blend ``src`` onto ``region`` (both ``(h, w, 4)`` bytes) in place -
    a source pixel of the mask color (all zero) leaves the backdrop as is."""
    s = np.ascontiguousarray(src).reshape(-1, 4)
    keep = s.view(np.uint32)[:, 0] != 0          # (not the mask color: all four zero)
    normal = mode is BlendMode.NORMAL or (grayscale and mode in _r._NON_SEPARABLE) or (
        mode not in _r._SEPARABLE and mode not in _r._NON_SEPARABLE)
    # (a cel's region is a slice of the canvas: reshaping copies it - written back below)
    flat = region.reshape(-1, 4).copy()
    changed = False
    if normal and opacity == 255:
        # Normal at full opacity: an opaque source pixel *is* the result
        # (sa = 255: out_a = 255, each channel d + (s - d) exactly) - copied
        opaque = keep & (s[:, 3] == 255)
        if opaque.any():
            flat[opaque] = s[opaque]
            keep &= ~opaque
            changed = True
    if keep.any():
        d = flat[keep].astype(np.int32)
        flat[keep] = blend(mode, new_blend, grayscale, d, s[keep].astype(np.int32),
                           opacity).astype(np.uint8)
        changed = True
    if changed:
        region[...] = flat.reshape(region.shape)


def blit_cel(sprite: Sprite, layer: Layer, cel: Cel, dest: bytearray) -> None:
    """:func:`aseprite._render._blit_cel`."""
    pixels = _r._cel_pixels(sprite, layer, cel)
    if pixels is None:
        return
    opacity = _r._mul_un8(_r._layer_opacity(sprite, layer), cel.opacity)
    x0, y0 = max(0, -cel.x), max(0, -cel.y)
    x1 = min(pixels.width, sprite.width - cel.x)
    y1 = min(pixels.height, sprite.height - cel.y)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba(sprite, layer, pixels)[y0:y1, x0:x1]
    region = _canvas(dest, sprite)[cel.y + y0: cel.y + y1, cel.x + x0: cel.x + x1]
    _blend_region(region, src, layer.blend_mode, sprite.new_blend,
                  sprite.color_mode is ColorMode.GRAYSCALE, opacity)


def blend_buffer(dest: bytearray, src: bytearray, opacity: int, blend_mode: BlendMode,
                 new_blend: bool = True, grayscale: bool = False) -> None:
    """:func:`aseprite._render._blend_buffer` (an isolated group onto its backdrop)."""
    d = np.frombuffer(dest, dtype=np.uint8).reshape(-1, 1, 4)
    s = np.frombuffer(bytes(src), dtype=np.uint8).reshape(-1, 1, 4)
    _blend_region(d, s, blend_mode, new_blend, grayscale, opacity)


def paint_indexed(indices: bytearray, painted: bytearray, width: int, height: int,
                  pixels: Pixels, cel: Cel, skip: int | None) -> None:
    """One cel of :func:`aseprite._render._flatten_indexed`: its indices over
    the canvas's - but the transparent one (``skip``)."""
    x0, y0 = max(0, -cel.x), max(0, -cel.y)
    x1 = min(pixels.width, width - cel.x)
    y1 = min(pixels.height, height - cel.y)
    if x1 <= x0 or y1 <= y0:
        return
    data = np.frombuffer(pixels.data, dtype=np.uint8)[: pixels.width * pixels.height]
    src = data.reshape(pixels.height, pixels.width)[y0:y1, x0:x1]
    idx = np.frombuffer(indices, dtype=np.uint8).reshape(height, width)
    pnt = np.frombuffer(painted, dtype=np.uint8).reshape(height, width)
    rows = slice(cel.y + y0, cel.y + y1)
    cols = slice(cel.x + x0, cel.x + x1)
    put = np.ones(src.shape, dtype=bool) if skip is None else src != skip
    idx[rows, cols][put] = src[put]
    pnt[rows, cols][put] = 0xFF
