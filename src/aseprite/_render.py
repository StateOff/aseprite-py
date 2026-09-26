"""Composite a frame to RGBA8."""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from aseprite._limits import (
    MAX_GROUP_DEPTH,
    MAX_PIXELS,
    MAX_UNCOMPRESSED_BYTES,
    bytes_per_tile,
)
from aseprite._model import (
    BlendMode,
    Cel,
    ColorMode,
    Layer,
    LayerType,
    Palette,
    Pixels,
)

if TYPE_CHECKING:
    from aseprite._sprite import Sprite


def flatten_frame(sprite: Sprite, frame_index: int) -> bytes:
    """Returns RGBA8 bytes for one composited frame."""
    if frame_index < 0 or frame_index >= len(sprite.frames):
        raise IndexError(f"frame {frame_index} is out of range")
    if sprite.width * sprite.height > MAX_PIXELS:
        raise ValueError(f"canvas exceeds {MAX_PIXELS} pixels")
    top_level = [layer for layer in sprite.layers if layer.child_level == 0]
    if sprite.color_mode is ColorMode.INDEXED:
        return _flatten_indexed(sprite, frame_index, top_level)
    dest = bytearray(sprite.width * sprite.height * 4)
    isolate_groups = sprite.group_blend
    scratches: list[bytearray] = []
    _composite_layers(sprite, frame_index, top_level, dest, isolate_groups, scratches)
    return bytes(dest)


def _flatten_indexed(sprite: Sprite, frame_index: int, layers: list[Layer]) -> bytes:
    """Composites an indexed sprite the way Aseprite does.

    Indexed sprites are composited in index space: a pixel replaces the one
    below it unless it is the transparent index on a non-background layer.
    Layer and cel opacity and group isolation do not apply. The palette is
    applied once at the end.
    """
    width, height = sprite.width, sprite.height
    indices = bytearray(width * height)
    painted = bytearray(width * height)  # 0xFF where a pixel was painted
    for _order, _z, layer, cel in _collect_entries(
        sprite, frame_index, layers, False, 0
    ):
        if cel is None:
            continue
        pixels = _cel_pixels(sprite, layer, cel)
        if pixels is None or pixels.color_mode is not ColorMode.INDEXED:
            continue
        skip = None if layer.background else sprite.transparent_index
        x0 = max(0, -cel.x)
        y0 = max(0, -cel.y)
        x1 = min(pixels.width, width - cel.x)
        y1 = min(pixels.height, height - cel.y)
        data = pixels.data
        for py in range(y0, y1):
            row = py * pixels.width
            dest_row = (cel.y + py) * width + cel.x
            for px in range(x0, x1):
                index = data[row + px]
                if index == skip:
                    continue
                indices[dest_row + px] = index
                painted[dest_row + px] = 0xFF
    return _apply_palette(sprite.palette_at(frame_index), indices, painted)


def _apply_palette(colors: Palette, indices: bytearray, painted: bytearray) -> bytes:
    """Maps palette indices to RGBA bytes without a per-pixel Python loop."""
    palette = colors.colors
    tables = []
    for channel in range(4):
        table = bytearray(256)
        for index, color in enumerate(palette[:256]):
            table[index] = (color.r, color.g, color.b, color.a)[channel]
        tables.append(bytes(table))
    total = len(indices)
    out = bytearray(total * 4)
    step = 1 << 16
    for start in range(0, total, step):
        chunk = bytes(indices[start : start + step])
        mask = int.from_bytes(painted[start : start + step], "big")
        for channel in range(4):
            values = int.from_bytes(chunk.translate(tables[channel]), "big") & mask
            out[start * 4 + channel : (start + len(chunk)) * 4 : 4] = values.to_bytes(
                len(chunk), "big"
            )
    return bytes(out)


def _composite_layers(
    sprite: Sprite,
    frame_index: int,
    layers: list[Layer],
    dest: bytearray,
    isolate_groups: bool,
    scratches: list[bytearray],
    depth: int = 0,
) -> None:
    entries = _collect_entries(sprite, frame_index, layers, isolate_groups, depth)
    for _order, _z, layer, cel in entries:
        if cel is None:
            _composite_group(
                sprite, frame_index, layer, dest, isolate_groups, scratches, depth
            )
        else:
            _blit_cel(sprite, layer, cel, dest)


def _collect_entries(
    sprite: Sprite,
    frame_index: int,
    layers: list[Layer],
    isolate_groups: bool,
    depth: int,
) -> list[tuple[int, int, Layer, Cel | None]]:
    """Returns the visible cels under ``layers`` in paint order.

    Aseprite sorts cels by ``layer index + z-index`` across the whole layer
    list, so a z-index can move a cel past a group boundary. When groups are
    isolated, each group is one entry and its children are sorted on their
    own.
    """
    if depth > MAX_GROUP_DEPTH:
        raise ValueError(f"layer group nesting exceeds {MAX_GROUP_DEPTH} levels")
    entries: list[tuple[int, int, Layer, Cel | None]] = []
    for layer in layers:
        if not layer.visible or layer.reference:
            continue
        if layer.kind is LayerType.GROUP:
            if isolate_groups:
                entries.append((layer.index, 0, layer, None))
            else:
                entries.extend(
                    _collect_entries(
                        sprite,
                        frame_index,
                        sprite.layers.children(layer),
                        isolate_groups,
                        depth + 1,
                    )
                )
            continue
        cel = _resolve_cel(sprite, layer, frame_index)
        if cel is None:
            continue
        entries.append((layer.index + cel.z_index, cel.z_index, layer, cel))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries


def _composite_group(
    sprite: Sprite,
    frame_index: int,
    group: Layer,
    dest: bytearray,
    isolate_groups: bool,
    scratches: list[bytearray],
    depth: int,
) -> None:
    children = sprite.layers.children(group)
    if not isolate_groups:
        _composite_layers(
            sprite, frame_index, children, dest, isolate_groups, scratches, depth + 1
        )
        return
    canvas_bytes = sprite.width * sprite.height * 4
    if (depth + 1) * canvas_bytes > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("isolated group compositing exceeds the size limit")
    while len(scratches) <= depth:
        scratches.append(bytearray(canvas_bytes))
    child_buf = scratches[depth]
    child_buf[:] = b"\x00" * canvas_bytes
    _composite_layers(
        sprite, frame_index, children, child_buf, isolate_groups, scratches, depth + 1
    )
    _blend_buffer(
        dest,
        child_buf,
        _layer_opacity(sprite, group),
        group.blend_mode,
        sprite.new_blend,
        sprite.color_mode is ColorMode.GRAYSCALE,
    )


def _layer_opacity(sprite: Sprite, layer: Layer) -> int:
    if not sprite.valid_layer_opacity or layer.background:
        return 255
    return layer.opacity


def _resolve_cel(sprite: Sprite, layer: Layer, frame_index: int) -> Cel | None:
    original = sprite.frames[frame_index].cel(layer)
    seen: set[int] = set()
    current = frame_index
    while current not in seen:
        seen.add(current)
        if current < 0 or current >= len(sprite.frames):
            return None
        cel = sprite.frames[current].cel(layer)
        if cel is None:
            return None
        if cel.link is None:
            if original is None or original is cel:
                return cel
            return replace(original, pixels=cel.pixels, tilemap=cel.tilemap, link=None)
        current = cel.link
    return None


def _cel_pixels(sprite: Sprite, layer: Layer, cel: Cel) -> Pixels | None:
    if cel.pixels is not None:
        return cel.pixels
    if cel.tilemap is None:
        return None
    return _tiles_to_pixels(sprite, layer, cel)


def _tiles_to_pixels(sprite: Sprite, layer: Layer, cel: Cel) -> Pixels | None:
    tm = cel.tilemap
    if tm is None or layer.tileset_index is None:
        return None
    # The layer field stores the file's tileset ID, which need not match
    # its position in the tileset chunk list.
    tileset = next((ts for ts in sprite.tilesets if ts.id == layer.tileset_index), None)
    if tileset is None:
        return None
    bpp = sprite.color_mode.bytes_per_pixel
    tw, th = tileset.tile_width, tileset.tile_height
    out_w = tm.width * tw
    out_h = tm.height * th
    if out_w < 0 or out_h < 0 or out_w * out_h > MAX_PIXELS:
        raise ValueError(f"tilemap exceeds {MAX_PIXELS} pixels")
    if sprite.color_mode is ColorMode.INDEXED:
        # Empty and out-of-range tiles are transparent, which in indexed
        # mode means the transparent index rather than index 0.
        out = bytearray((sprite.transparent_index,)) * (out_w * out_h)
        empty_pixel = bytes((sprite.transparent_index,))
    else:
        out = bytearray(out_w * out_h * bpp)
        empty_pixel = bytes(bpp)
    tile_bytes = tw * th * bpp
    stride = bytes_per_tile(tm.bits_per_tile)
    id_mask = tm.tile_id_mask
    id_shift = (id_mask & -id_mask).bit_length() - 1 if id_mask else 0
    pixel_data = tileset.pixels.data if tileset.pixels is not None else b""
    for ty in range(tm.height):
        for tx in range(tm.width):
            offset = (ty * tm.width + tx) * stride
            if offset + stride > len(tm.tiles):
                continue
            if stride == 4:
                value = struct.unpack_from("<I", tm.tiles, offset)[0]
            elif stride == 2:
                value = struct.unpack_from("<H", tm.tiles, offset)[0]
            else:
                value = tm.tiles[offset]
            tile_id = (value & id_mask) >> id_shift
            # Aseprite draws tile 0 like any other tile; it is empty only
            # because the editor keeps that tile's image transparent.
            src = tile_id * tile_bytes
            if src + tile_bytes > len(pixel_data):
                continue
            tile = pixel_data[src : src + tile_bytes]
            x_flip = bool(value & tm.x_flip_mask)
            y_flip = bool(value & tm.y_flip_mask)
            d_flip = bool(value & tm.d_flip_mask)
            tile, fw, fh = _flip_tile(
                bytes(tile), tw, th, bpp, x_flip, y_flip, d_flip, empty_pixel
            )
            for row in range(fh):
                dest_y = ty * th + row
                dest_x = tx * tw
                if dest_y < 0 or dest_y >= out_h:
                    continue
                dest_row = (dest_y * out_w + dest_x) * bpp
                src_row = row * fw * bpp
                copy = min(fw, out_w - dest_x) * bpp
                if copy > 0:
                    out[dest_row : dest_row + copy] = tile[src_row : src_row + copy]
    return Pixels(out_w, out_h, bytes(out), sprite.color_mode)


def _flip_tile(
    data: bytes,
    width: int,
    height: int,
    bpp: int,
    x_flip: bool,
    y_flip: bool,
    d_flip: bool,
    empty_pixel: bytes,
) -> tuple[bytes, int, int]:
    if not (x_flip or y_flip or d_flip):
        return data, width, height
    out = bytearray(empty_pixel * (width * height))
    # Aseprite flips inside the original tile footprint. Transposed
    # coordinates outside a rectangular tile are transparent, not spilled
    # into the next tile. Invert X/Y flips before looking up the transpose.
    for y in range(height):
        for x in range(width):
            sx = width - 1 - x if x_flip else x
            sy = height - 1 - y if y_flip else y
            if d_flip:
                sx, sy = sy, sx
            if sx < width and sy < height:
                src = (sy * width + sx) * bpp
                dest = (y * width + x) * bpp
                out[dest : dest + bpp] = data[src : src + bpp]
    return bytes(out), width, height


def _blit_cel(sprite: Sprite, layer: Layer, cel: Cel, dest: bytearray) -> None:
    pixels = _cel_pixels(sprite, layer, cel)
    if pixels is None:
        return
    opacity = _mul_un8(_layer_opacity(sprite, layer), cel.opacity)
    # an image / tilemap layer's blend mode always applies (file spec NOTE.6)
    blender = get_blender(
        layer.blend_mode, sprite.new_blend, sprite.color_mode is ColorMode.GRAYSCALE
    )
    # Only visit the part of the cel that lands on the canvas, so the cost
    # is bounded by the canvas size rather than the cel size.
    x0 = max(0, -cel.x)
    y0 = max(0, -cel.y)
    x1 = min(pixels.width, sprite.width - cel.x)
    y1 = min(pixels.height, sprite.height - cel.y)
    for py in range(y0, y1):
        dy = cel.y + py
        for px in range(x0, x1):
            dx = cel.x + px
            src = _pixel_rgba(sprite, layer, pixels, px, py)
            if src == _MASK:
                continue  # Aseprite's BlenderHelper leaves the backdrop as is
            di = (dy * sprite.width + dx) * 4
            dest[di : di + 4] = blender(bytes(dest[di : di + 4]), src, opacity)


def _pixel_rgba(sprite: Sprite, layer: Layer, pixels: Pixels, x: int, y: int) -> bytes:
    i = (y * pixels.width + x) * pixels.color_mode.bytes_per_pixel
    if pixels.color_mode is ColorMode.RGBA:
        return bytes(pixels.data[i : i + 4])
    if pixels.color_mode is ColorMode.GRAYSCALE:
        value, alpha = pixels.data[i], pixels.data[i + 1]
        return bytes((value, value, value, alpha))
    index = pixels.data[i]
    if not layer.background and index == sprite.transparent_index:
        return b"\x00\x00\x00\x00"
    if index < len(sprite.palette):
        color = sprite.palette[index]
        return bytes((color.r, color.g, color.b, color.a))
    return b"\x00\x00\x00\x00"


def _blend_buffer(
    dest: bytearray,
    src: bytearray,
    opacity: int,
    blend_mode: BlendMode,
    new_blend: bool = True,
    grayscale: bool = False,
) -> None:
    """Composites an isolated group's buffer ``src`` onto ``dest``.

    A group's blend mode and opacity are valid when the header's group
    blend flag is set (file spec NOTE.6) - the only case groups are
    isolated.
    """
    blender = get_blender(blend_mode, new_blend, grayscale)
    for i in range(0, len(dest), 4):
        pixel = bytes(src[i : i + 4])
        if pixel != _MASK:
            dest[i : i + 4] = blender(bytes(dest[i : i + 4]), pixel, opacity)


#: the mask color of RGBA and grayscale images: a source pixel equal to it
#: is skipped - the backdrop keeps its bytes (Aseprite's ``BlenderHelper``)
_MASK = b"\x00\x00\x00\x00"


def _mul_un8(a: int, b: int) -> int:
    """Returns ``a * b / 255`` rounded the way Aseprite's ``MUL_UN8`` does."""
    t = a * b + 0x80
    return ((t >> 8) + t) >> 8


def _blend_normal(dst: bytes, src: bytes, opacity: int) -> bytes:
    """Blends ``src`` over ``dst`` with Aseprite's ``rgba_blender_normal``.

    The arithmetic matches the editor so that flattened pixels agree with
    its own export, including the truncating division on each channel.
    """
    if dst[3] == 0:
        return bytes((src[0], src[1], src[2], _mul_un8(src[3], opacity)))
    if src[3] == 0:
        return dst
    sa = _mul_un8(src[3], opacity)
    da = dst[3]
    out_a = sa + da - _mul_un8(da, sa)
    out = bytearray(4)
    for c in range(3):
        delta = (src[c] - dst[c]) * sa
        # C integer division truncates toward zero.
        step = -(-delta // out_a) if delta < 0 else delta // out_a
        out[c] = dst[c] + step
    out[3] = out_a
    return bytes(out)


# ---------------------------------------------------------------------------
# Blend modes - ported from Aseprite's ``src/doc/blend_funcs.cpp`` so the
# composited bytes match its own export: the same 8-bit integer arithmetic
# (pixman's ``MUL_UN8`` / ``DIV_UN8``), the same floating point for soft
# light and the HSL modes, and the same final composite through
# ``rgba_blender_normal``.
# ---------------------------------------------------------------------------


def _div_un8(a: int, b: int) -> int:
    """Returns ``a * 255 / b`` rounded the way pixman's ``DIV_UN8`` does."""
    return (a * 0xFF + b // 2) // b


def _multiply(b: int, s: int) -> int:
    return _mul_un8(b, s)


def _screen(b: int, s: int) -> int:
    return b + s - _mul_un8(b, s)


def _hard_light(b: int, s: int) -> int:
    return _multiply(b, s << 1) if s < 128 else _screen(b, (s << 1) - 255)


def _overlay(b: int, s: int) -> int:
    return _hard_light(s, b)


def _darken(b: int, s: int) -> int:
    return min(b, s)


def _lighten(b: int, s: int) -> int:
    return max(b, s)


def _color_dodge(b: int, s: int) -> int:
    if b == 0:
        return 0
    s = 255 - s
    return 255 if b >= s else _div_un8(b, s)


def _color_burn(b: int, s: int) -> int:
    if b == 255:
        return 255
    b = 255 - b
    return 0 if b >= s else 255 - _div_un8(b, s)


def _soft_light(b8: int, s8: int) -> int:
    b = b8 / 255.0
    s = s8 / 255.0
    d = ((16 * b - 12) * b + 4) * b if b <= 0.25 else b**0.5
    if s <= 0.5:
        r = b - (1.0 - 2.0 * s) * b * (1.0 - b)
    else:
        r = b + (2.0 * s - 1.0) * (d - b)
    return int(r * 255 + 0.5)


def _difference(b: int, s: int) -> int:
    return abs(b - s)


def _exclusion(b: int, s: int) -> int:
    return b + s - 2 * _mul_un8(b, s)


def _addition(b: int, s: int) -> int:
    return min(b + s, 255)


def _subtract(b: int, s: int) -> int:
    return max(b - s, 0)


def _divide(b: int, s: int) -> int:
    if b == 0:
        return 0
    return 255 if b >= s else _div_un8(b, s)


_SEPARABLE = {
    BlendMode.MULTIPLY: _multiply,
    BlendMode.SCREEN: _screen,
    BlendMode.OVERLAY: _overlay,
    BlendMode.DARKEN: _darken,
    BlendMode.LIGHTEN: _lighten,
    BlendMode.COLOR_DODGE: _color_dodge,
    BlendMode.COLOR_BURN: _color_burn,
    BlendMode.HARD_LIGHT: _hard_light,
    BlendMode.SOFT_LIGHT: _soft_light,
    BlendMode.DIFFERENCE: _difference,
    BlendMode.EXCLUSION: _exclusion,
    BlendMode.ADDITION: _addition,
    BlendMode.SUBTRACT: _subtract,
    BlendMode.DIVIDE: _divide,
}


def _lum(r: float, g: float, b: float) -> float:
    return 0.3 * r + 0.59 * g + 0.11 * b


def _sat(r: float, g: float, b: float) -> float:
    return max(r, g, b) - min(r, g, b)


def _clip_color(r: float, g: float, b: float) -> tuple[float, float, float]:
    lum = _lum(r, g, b)
    n = min(r, g, b)
    x = max(r, g, b)
    if n < 0:
        r = lum + (((r - lum) * lum) / (lum - n))
        g = lum + (((g - lum) * lum) / (lum - n))
        b = lum + (((b - lum) * lum) / (lum - n))
    if x > 1:
        r = lum + (((r - lum) * (1 - lum)) / (x - lum))
        g = lum + (((g - lum) * (1 - lum)) / (x - lum))
        b = lum + (((b - lum) * (1 - lum)) / (x - lum))
    return r, g, b


def _set_lum(r: float, g: float, b: float, lum: float) -> tuple[float, float, float]:
    d = lum - _lum(r, g, b)
    return _clip_color(r + d, g + d, b + d)


def _set_sat(r: float, g: float, b: float, sat: float) -> tuple[float, float, float]:
    lo = min(r, g, b)
    span = max(r, g, b) - lo
    if span > 0.0:
        return ((r - lo) * sat) / span, ((g - lo) * sat) / span, ((b - lo) * sat) / span
    return 0.0, 0.0, 0.0


def _unit(c: bytes) -> tuple[float, float, float]:
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0


def _hsl_hue(dst: bytes, src: bytes) -> tuple[float, float, float]:
    br, bg, bb = _unit(dst)
    r, g, b = _set_sat(*_unit(src), _sat(br, bg, bb))
    return _set_lum(r, g, b, _lum(br, bg, bb))


def _hsl_saturation(dst: bytes, src: bytes) -> tuple[float, float, float]:
    br, bg, bb = _unit(dst)
    r, g, b = _set_sat(br, bg, bb, _sat(*_unit(src)))
    return _set_lum(r, g, b, _lum(br, bg, bb))


def _hsl_color(dst: bytes, src: bytes) -> tuple[float, float, float]:
    return _set_lum(*_unit(src), _lum(*_unit(dst)))


def _hsl_luminosity(dst: bytes, src: bytes) -> tuple[float, float, float]:
    return _set_lum(*_unit(dst), _lum(*_unit(src)))


_NON_SEPARABLE = {
    BlendMode.HUE: _hsl_hue,
    BlendMode.SATURATION: _hsl_saturation,
    BlendMode.COLOR: _hsl_color,
    BlendMode.LUMINOSITY: _hsl_luminosity,
}


def _channel(value: float) -> int:
    """A C ``int`` cast of ``255 * value`` (truncation), kept in a byte."""
    return min(255, max(0, int(255.0 * value)))


def _blend_merge(dst: bytes, src: bytes, opacity: int) -> bytes:
    """Aseprite's ``rgba_blender_merge``: ``dst`` moved toward ``src``."""
    if dst[3] == 0:
        rgb = src[:3]
    elif src[3] == 0:
        rgb = dst[:3]
    else:
        rgb = bytes(dst[c] + _mul_un8(src[c] - dst[c], opacity) for c in range(3))
    alpha = dst[3] + _mul_un8(src[3] - dst[3], opacity)
    if alpha == 0:
        rgb = b"\x00\x00\x00"
    return bytes((*rgb, alpha))


def _classic(mode: BlendMode) -> Callable[[bytes, bytes, int], bytes]:
    """The mode's color, then composited as Normal - Aseprite's
    ``rgba_blender_<mode>``."""
    separable = _SEPARABLE.get(mode)
    if separable is not None:

        def blend(dst: bytes, src: bytes, opacity: int) -> bytes:
            color = bytes(separable(dst[c], src[c]) for c in range(3))
            return _blend_normal(dst, color + src[3:4], opacity)

        return blend
    hsl = _NON_SEPARABLE[mode]

    def blend_hsl(dst: bytes, src: bytes, opacity: int) -> bytes:
        color = bytes(_channel(v) for v in hsl(dst, src))
        return _blend_normal(dst, color + src[3:4], opacity)

    return blend_hsl


def _new_blend(
    classic: Callable[[bytes, bytes, int], bytes],
) -> Callable[[bytes, bytes, int], bytes]:
    """Aseprite's ``RGBA_BLENDER_N``: the mode's result, faded toward Normal
    by how transparent the backdrop is."""

    def blend(dst: bytes, src: bytes, opacity: int) -> bytes:
        if dst[3] == 0:
            return _blend_normal(dst, src, opacity)
        normal = _blend_normal(dst, src, opacity)
        mode = classic(dst, src, opacity)
        merged = _blend_merge(normal, mode, dst[3])
        composite = _mul_un8(dst[3], _mul_un8(src[3], opacity))
        return _blend_merge(merged, mode, composite)

    return blend


_BLENDERS: dict[tuple[BlendMode, bool], Callable[[bytes, bytes, int], bytes]] = {}


def get_blender(
    mode: BlendMode, new_blend: bool = True, grayscale: bool = False
) -> Callable[[bytes, bytes, int], bytes]:
    """Returns the pixel blender for ``mode``: ``(dst, src, opacity) -> RGBA``.

    ``new_blend`` picks Aseprite's "new blending" variant of the non-Normal
    modes (its default since 1.3), which fades a mode toward Normal over a
    semi-transparent backdrop. Unknown modes blend as Normal.

    ``grayscale`` follows Aseprite's grayscale blenders (``get_graya_blender``):
    the HSL modes blend as Normal, and Addition with new blending uses the
    Exclusion blender - as the editor does, so exports agree.
    """
    if grayscale:
        if mode in _NON_SEPARABLE:
            return _blend_normal
        if mode is BlendMode.ADDITION and new_blend:
            mode = BlendMode.EXCLUSION
    if mode is BlendMode.NORMAL or (
        mode not in _SEPARABLE and mode not in _NON_SEPARABLE
    ):
        return _blend_normal
    key = (mode, new_blend)
    blender = _BLENDERS.get(key)
    if blender is None:
        classic = _classic(mode)
        blender = _new_blend(classic) if new_blend else classic
        _BLENDERS[key] = blender
    return blender
