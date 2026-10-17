#!/usr/bin/env python3
"""AvianVisitors - cut the cream ground off the generated illustrations.

Step 2 of the illustration pipeline (after pregen.py, before build_masks.py).

pregen.py renders each bird on a flat cream ground because the image model
can't cut a clean transparent background on its own (it leaves holes and
fringes). A flat known ground, by contrast, removes cleanly. This runs each
illustration through the BiRefNet matting model (via rembg), fills back any
pale plumage the matte mistook for ground (see matte()), then crops to the
bird's bounding box with a small even margin, and saves an RGBA cutout back
in place.

Idempotent: an illustration that already has a transparent background is
skipped unless you pass --force.

Requires rembg + onnxruntime (see requirements.txt). The first run downloads
the BiRefNet model (~1 GB) to ~/.u2net/.

Usage:
    python3 cutout.py                      # process every illustration
    python3 cutout.py calypte-anna         # one slug (both poses)
    python3 cutout.py calypte-anna-2 --force
    python3 cutout.py hirundo-neoxena --force --close 6   # belly outline has gaps
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import build_masks

CREAM_TOL = 11   # near-paper distance
PLUMAGE = 18     # beyond this distance from paper counts as inked body


def matte(im, session, remove, close=0):
    """BiRefNet matte + exterior-cream peel + belly fill, as an RGBA image.

    BiRefNet alone reads white and cream plumage (albatross bodies, gull
    necks, penguin bellies) as ground and cuts it out, leaving the bird an
    outline. Only cream connected to the frame edge is ground, so enclosed
    pale regions are filled back in. A filled region whose boundary is mostly
    not plumage is a between-legs pocket, not a belly, and stays clear.

    close > 0 also bridges outline gaps up to about 2*close px before the
    fill, for a belly whose outline is broken and so opens onto the ground.
    Opt-in per bird: a large radius also fills between tail streamers and
    legs, and on flight poses between wing and body."""
    import numpy as np
    from PIL import Image, ImageFilter
    from scipy import ndimage

    rgb = im.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    a = np.asarray(remove(rgb, session=session).getchannel("A"))
    corners = np.concatenate([arr[:15, :15].reshape(-1, 3), arr[:15, -15:].reshape(-1, 3),
                              arr[-15:, :15].reshape(-1, 3), arr[-15:, -15:].reshape(-1, 3)])
    paper = np.median(corners, axis=0)
    dist = np.sqrt(((arr - paper) ** 2).sum(2))
    passable = (a < 100) | (dist < CREAM_TOL)
    lbl, _n = ndimage.label(passable)
    border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border.discard(0)
    exterior = np.isin(lbl, list(border))
    base = (a >= 100) & ~exterior
    if close > 0:
        y, x = np.ogrid[-close:close + 1, -close:close + 1]
        disk = x * x + y * y <= close * close
        solid = ndimage.binary_fill_holes(base | ndimage.binary_closing(base, structure=disk))
    else:
        solid = ndimage.binary_fill_holes(base)
    added = solid & ~base
    plumage = (a >= 100) & (dist > PLUMAGE)
    al, an = ndimage.label(added)
    reject = np.zeros_like(solid)
    for i in range(1, an + 1):
        region = al == i
        ring = ndimage.binary_dilation(region, iterations=3) & ~region & ~exterior
        if ring.sum() == 0 or plumage[ring].mean() < 0.30:
            reject |= region
    solid = solid & ~reject
    labels, m = ndimage.label(solid)
    if m > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, m + 1))
        solid = labels == int(np.argmax(sizes)) + 1
    inside = ndimage.binary_erosion(solid, iterations=2)
    af = a.copy()
    af[inside] = 255
    af[~solid] = 0
    af = np.maximum(np.asarray(Image.fromarray(af).filter(ImageFilter.GaussianBlur(0.4))),
                    (inside * 255).astype(np.uint8))
    af[~solid] = 0
    return Image.fromarray(np.dstack([np.asarray(rgb), af]).astype(np.uint8), "RGBA")


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*",
                    help="Slugs to process (e.g. calypte-anna). Default: all.")
    ap.add_argument("--dir", type=Path, default=here / "assets" / "illustrations",
                    help="Illustration directory (default: avian/assets/illustrations/)")
    ap.add_argument("--model", default="birefnet-general",
                    help="rembg model name (default: birefnet-general)")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="Even margin around the bird, as a fraction of its "
                         "long side (default: 0.02)")
    ap.add_argument("--force", action="store_true",
                    help="Re-cut illustrations that already have transparency")
    ap.add_argument("--close", type=int, default=0, metavar="PX",
                    help="Bridge belly-outline gaps of up to ~2*PX pixels before "
                         "the fill (default 0; try 6 for a hollow belly)")
    args = ap.parse_args()

    try:
        from rembg import new_session, remove
        from image_files import open_source_image, save_png_atomic
        import numpy  # noqa: F401  (matte)
        import scipy.ndimage  # noqa: F401  (matte)
    except ImportError:
        print("error: needs Pillow + rembg + scipy (pip install -r requirements.txt)",
              file=sys.stderr)
        return 2

    if args.slugs:
        paths = []
        for slug in args.slugs:
            paths.append(args.dir / f"{slug}.png")
            flight = args.dir / f"{slug}-2.png"
            if not slug.endswith("-2") and flight.exists():
                paths.append(flight)
        missing = [p for p in paths if not p.exists()]
        if missing:
            print("error: not found: " + ", ".join(p.name for p in missing), file=sys.stderr)
            return 2
    else:
        paths = sorted(args.dir.glob("*.png"))
    if not paths:
        print("error: no illustrations found", file=sys.stderr)
        return 1

    session = new_session(args.model)
    done = skipped = 0
    for p in paths:
        original_identity = build_masks.regular_file_identity(p)
        with open_source_image(p.read_bytes()) as im:
            if not args.force and im.mode == "RGBA" and im.getchannel("A").getextrema()[0] == 0:
                skipped += 1
                continue
            cut = matte(im, session, remove, close=args.close)  # RGBA, ground -> transparent
        bbox = cut.getchannel("A").getbbox()
        if bbox:
            pad = round(args.margin * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            x0, y0 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
            x1, y1 = min(cut.width, bbox[2] + pad), min(cut.height, bbox[3] + pad)
            cut = cut.crop((x0, y0, x1, y1))
        try:
            with build_masks.included_art_write_transaction(
                args.dir, {p: original_identity}
            ):
                save_png_atomic(cut, p)
                build_masks.fsync_directory(p.parent)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        done += 1
        print(f"  [cut]  {p.name}  -> {cut.width}x{cut.height}")

    print(f"\ncut {done} · skipped {skipped} (already transparent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
