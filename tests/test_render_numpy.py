"""The NumPy compositing path gives the pure-Python path's bytes - every blend
mode, old and new blending, grayscale, every opacity class."""

import os
import random

import pytest

np = pytest.importorskip("numpy")

from aseprite import BlendMode, Color, ColorMode, Palette, Pixels, Sprite  # noqa: E402
from aseprite import _render  # noqa: E402
from aseprite import _render_np  # noqa: E402
from aseprite._render import get_blender  # noqa: E402


def _pixels(rng: random.Random, n: int) -> list[bytes]:
    """Random RGBA pixels, with the edge cases often: alpha 0 / 255, the
    mask color, black and white."""
    out = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.1:
            out.append(bytes(4))
        elif roll < 0.2:
            out.append(bytes((rng.randrange(256), rng.randrange(256), rng.randrange(256), 0)))
        elif roll < 0.4:
            out.append(bytes((rng.randrange(256), rng.randrange(256), rng.randrange(256), 255)))
        elif roll < 0.45:
            out.append(bytes((255, 255, 255, rng.randrange(256))))
        else:
            out.append(bytes(rng.randrange(256) for _ in range(4)))
    return out


@pytest.mark.parametrize("mode", list(BlendMode))
@pytest.mark.parametrize("new_blend", [True, False])
@pytest.mark.parametrize("grayscale", [False, True])
def test_blend_matches_the_scalar_blender(mode, new_blend, grayscale):
    rng = random.Random(f"{mode}-{new_blend}-{grayscale}")
    dst, src = _pixels(rng, 600), _pixels(rng, 600)
    scalar = get_blender(mode, new_blend, grayscale)
    for opacity in (0, 1, 77, 128, 254, 255):
        want = [scalar(d, s, opacity) for d, s in zip(dst, src)]
        d = np.array([list(p) for p in dst], dtype=np.int32)
        s = np.array([list(p) for p in src], dtype=np.int32)
        got = _render_np.blend(mode, new_blend, grayscale, d, s, opacity)
        assert [bytes(row.astype(np.uint8)) for row in got] == want, (mode, opacity)


def _layered(mode: BlendMode, color_mode: ColorMode = ColorMode.RGBA) -> Sprite:
    rng = random.Random(str(mode))
    s = Sprite(24, 20, color_mode, empty=True)
    if color_mode is ColorMode.INDEXED:
        s.palette = Palette([Color(rng.randrange(256), rng.randrange(256),
                                   rng.randrange(256), rng.choice((0, 128, 255)))
                             for _ in range(40)])
    bottom, top = s.add_layer("bottom"), s.add_layer("top")
    top.blend_mode = mode
    top.opacity = 200
    frame = s.add_frame(100)
    bpp = color_mode.bytes_per_pixel

    def data(w, h):
        if color_mode is ColorMode.INDEXED:
            return bytes(rng.randrange(40) for _ in range(w * h))
        return b"".join(_pixels(rng, w * h)) if bpp == 4 else bytes(
            rng.randrange(256) for _ in range(w * h * bpp))

    frame.set_cel(bottom, Pixels(20, 16, data(20, 16), color_mode), 2, 1)
    # partly off the canvas: only what lands on it is visited
    frame.set_cel(top, Pixels(16, 14, data(16, 14), color_mode), 14, -5)
    return s


@pytest.mark.parametrize("mode", list(BlendMode))
@pytest.mark.parametrize("color_mode", [ColorMode.RGBA, ColorMode.GRAYSCALE,
                                        ColorMode.INDEXED])
def test_a_flattened_frame_is_the_same_both_ways(mode, color_mode, monkeypatch):
    sprite = _layered(mode, color_mode)
    fast = _render.flatten_frame(sprite, 0)
    monkeypatch.setattr(_render, "_NUMPY", False)          # the pure-Python path
    assert _render._numpy() is None
    assert _render.flatten_frame(sprite, 0) == fast


def test_pure_python_when_asked(monkeypatch):
    monkeypatch.setattr(_render, "_NUMPY", None)
    monkeypatch.setenv("ASEPRITE_PURE_PYTHON", "1")
    assert _render._numpy() is None
    monkeypatch.setattr(_render, "_NUMPY", None)
    monkeypatch.delenv("ASEPRITE_PURE_PYTHON")
    assert _render._numpy() is _render_np
    assert os.environ.get("ASEPRITE_PURE_PYTHON") is None


@pytest.mark.parametrize("mode", [BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.HUE,
                                  BlendMode.SOFT_LIGHT])
def test_an_isolated_group_is_the_same_both_ways(mode, monkeypatch):
    from aseprite import LayerType

    rng = random.Random(f"group-{mode}")
    s = Sprite(16, 12, ColorMode.RGBA, empty=True)
    base = s.add_layer("base")
    group = s.add_layer("group", kind=LayerType.GROUP)
    group.blend_mode = mode
    group.opacity = 170
    inner = s.add_layer("inner", parent=group)
    inner.blend_mode = BlendMode.SCREEN
    over = s.add_layer("over", parent=group)
    frame = s.add_frame(100)
    for ly, (x, y) in ((base, (0, 0)), (inner, (3, 2)), (over, (-4, 5))):
        frame.set_cel(ly, Pixels(14, 10, b"".join(_pixels(rng, 140)), ColorMode.RGBA), x, y)
    s.group_blend = True
    fast = _render.flatten_frame(s, 0)
    monkeypatch.setattr(_render, "_NUMPY", False)
    assert _render.flatten_frame(s, 0) == fast
