"""cutout.matte keeps pale plumage that BiRefNet mistakes for the cream ground."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "avian" / "scripts"))
import cutout  # noqa: E402

CREAM = (240, 235, 225)
BELLY = (246, 243, 236)  # white plumage, within CREAM_TOL of the ground


def pale_bellied_bird():
    image = Image.new("RGB", (200, 200), CREAM)
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 160, 160), fill=(30, 40, 60))
    draw.ellipse((70, 70, 130, 130), fill=BELLY)
    return image


def hollowing_remove(image, session):
    """Stand-in for BiRefNet: keeps the dark body, drops the pale belly."""
    rgba = image.convert("RGBA")
    alpha = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.ellipse((40, 40, 160, 160), fill=255)
    draw.ellipse((70, 70, 130, 130), fill=0)
    rgba.putalpha(alpha)
    return rgba


def test_enclosed_pale_belly_is_filled_back_in():
    cut = cutout.matte(pale_bellied_bird(), object(), hollowing_remove)
    alpha = cut.getchannel("A")
    assert cut.mode == "RGBA"
    assert alpha.getpixel((100, 100)) == 255
    assert cut.getpixel((100, 100))[:3] == BELLY
    assert alpha.getpixel((50, 100)) == 255


def test_ground_outside_the_bird_stays_clear():
    cut = cutout.matte(pale_bellied_bird(), object(), hollowing_remove)
    alpha = cut.getchannel("A")
    for corner in ((0, 0), (199, 0), (0, 199), (199, 199), (20, 100)):
        assert alpha.getpixel(corner) == 0


def open_bellied_remove(image, session):
    """BiRefNet on a belly whose outline has a gap at the bottom: the pale
    belly leaks into the ground through the gap and is dropped."""
    rgba = hollowing_remove(image, session)
    alpha = rgba.getchannel("A")
    ImageDraw.Draw(alpha).rectangle((96, 125, 104, 165), fill=0)
    rgba.putalpha(alpha)
    return rgba


def open_bellied_bird():
    image = pale_bellied_bird()
    ImageDraw.Draw(image).rectangle((96, 125, 104, 165), fill=CREAM)
    return image


def test_close_bridges_a_broken_belly_outline():
    open_cut = cutout.matte(open_bellied_bird(), object(), open_bellied_remove)
    assert open_cut.getchannel("A").getpixel((100, 100)) == 0

    closed = cutout.matte(open_bellied_bird(), object(), open_bellied_remove, close=6)
    alpha = closed.getchannel("A")
    assert alpha.getpixel((100, 100)) == 255
    for corner in ((0, 0), (199, 199), (20, 100), (100, 190)):
        assert alpha.getpixel(corner) == 0
