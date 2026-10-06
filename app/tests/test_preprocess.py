"""Testes unitários do preprocess.py. Rodar a partir da pasta do código:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess as pp  # noqa: E402


def fundus(w=400, h=300, r=120, value=150):
    """Retina sintética: círculo colorido centralizado sobre fundo preto."""
    arr = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    arr[(xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= r ** 2] = (value, value // 2, value // 4)
    return Image.fromarray(arr)


def bright_extent(im):
    """Largura e altura da região de retina numa imagem já processada."""
    mask = pp.to_array(im).max(axis=2) > pp.BRIGHTNESS_TOL
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    return cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1


def test_output_is_square_rgb_uint8():
    out, found = pp.preprocess_image(fundus(), 224)
    arr = pp.to_array(out)
    assert found and arr.shape == (224, 224, 3) and arr.dtype == np.uint8


def test_black_borders_are_removed():
    box = pp.find_retina_bbox(fundus(w=400, h=300, r=120))
    assert box == pytest.approx((80, 30, 321, 271), abs=6)


def test_retina_fills_output():
    out, _ = pp.preprocess_image(fundus(w=600, h=300, r=100), 224)
    w, h = bright_extent(out)
    assert w == pytest.approx(224, abs=4) and h == pytest.approx(224, abs=4)


def test_truncated_circle_is_padded_not_stretched():
    # círculo cortado em cima e embaixo (comum no APTOS): diâmetro 300 > altura 200
    out, found = pp.preprocess_image(fundus(w=400, h=200, r=150), 300)
    w, h = bright_extent(out)
    assert found
    assert w == pytest.approx(300, abs=6)   # largura ocupa a imagem toda
    assert h == pytest.approx(200, abs=6)   # altura mantém a proporção original, com faixas pretas


def test_stray_bright_pixels_are_ignored():
    im = fundus()
    arr = np.array(im)
    arr[5, 5] = 255
    arr[290, 395] = 255
    assert pp.find_retina_bbox(Image.fromarray(arr)) == pp.find_retina_bbox(im)


def test_dark_retina_is_still_found():
    assert pp.find_retina_bbox(fundus(value=40)) is not None


def test_black_image_returns_not_found():
    out, found = pp.preprocess_image(Image.new("RGB", (300, 300)), 224)
    assert not found and out.size == (224, 224)


def test_tiny_retina_returns_not_found():
    assert pp.find_retina_bbox(fundus(w=400, h=300, r=20)) is None


def test_accepts_grayscale_and_rgba():
    for mode in ("L", "RGBA"):
        out, found = pp.preprocess_image(fundus().convert(mode), 224)
        assert out.mode == "RGB" and found


def test_is_deterministic():
    im = fundus()
    a = pp.to_array(pp.preprocess_image(im, 224)[0])
    b = pp.to_array(pp.preprocess_image(im, 224)[0])
    assert np.array_equal(a, b)
