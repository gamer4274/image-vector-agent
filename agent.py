#!/usr/bin/env python3
"""Agent analyste d'image : JPEG -> tracé vectoriel SVG + DXF.

Usage:
    python agent.py entree.jpg -o sortie/ [--mode auto|centerline|outline]
                    [--quality max|high|balanced] [--unit mm|cm|px] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

import ezdxf

QUALITY_TOLERANCE = {"max": 0.45, "high": 0.8, "balanced": 1.6}
TARGET_MIN_SIDE = 1600
MAX_UPSCALE = 4.0

INSUNITS = {"px": 0, "in": 1, "mm": 4, "cm": 5}


@dataclass
class Analysis:
    width: int
    height: int
    contrast_std: float
    blur_variance: float
    noise_sigma: float
    ink_fraction: float
    ink_polarity: str
    threshold_method: str
    stroke_width_px: float
    mode_used: str


@dataclass
class VectorPath:
    points: list
    closed: bool


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return img


def preprocess(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    max_side = max(h, w)
    scale = 1.0
    if max_side < TARGET_MIN_SIDE:
        scale = min(MAX_UPSCALE, TARGET_MIN_SIDE / max_side)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_sigma = float(np.median(np.abs(lap)) / 0.6745)
    if noise_sigma > 8.0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=min(15.0, noise_sigma * 0.7), templateWindowSize=7, searchWindowSize=21)

    if float(gray.std()) < 40.0:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray, scale


def binarize(gray: np.ndarray, method: str = "auto", invert: bool = False) -> tuple[np.ndarray, str]:
    gray_work = 255 - gray if invert else gray
    otsu_t, otsu_mask = cv2.threshold(gray_work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_frac = float(np.count_nonzero(otsu_mask)) / otsu_mask.size

    block = max(31, (min(gray_work.shape) // 20) | 1)
    inv = cv2.adaptiveThreshold(gray_work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)

    otsu_ok = 0.004 <= otsu_frac <= 0.72 and float(gray_work.std()) > 35.0
    if method == "otsu" or (method == "auto" and otsu_ok):
        return otsu_mask, "otsu"
    if method == "adaptive":
        return inv, "adaptive"

    otsu_mask = otsu_mask if otsu_ok else inv
    return otsu_mask, "otsu" if otsu_ok else "adaptive"


def remove_specks(mask: np.ndarray, min_area_frac: float = 2e-5) -> np.ndarray:
    n = np.count_nonzero(mask)
    min_area = max(6.0, min_area_frac * mask.size)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def estimate_stroke_width(mask: np.ndarray) -> float:
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[dist > 0.4]
    if vals.size == 0:
        return 1.0
    return float(2.0 * np.median(vals))


def trace_skeleton(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    pts = set(map(tuple, np.argwhere(skel)))
    if not pts:
        return []
    neighbors = {}
    for y, x in pts:
        lst = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pts:
                    lst.append(q)
        neighbors[(y, x)] = lst

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    used = set()
    nodes = {p for p in pts if len(neighbors[p]) != 2}
    paths = []

    def walk(start, first):
        path = [start, first]
        used.add(ekey(start, first))
        prev, cur = start, first
        while cur not in nodes:
            nxt = [q for q in neighbors[cur] if q != prev and ekey(cur, q) not in used]
            if not nxt:
                break
            q = nxt[0]
            used.add(ekey(cur, q))
            path.append(q)
            prev, cur = cur, q
            if len(path) > len(pts) + 1:
                break
        return path

    for n in sorted(nodes):
        for m in neighbors[n]:
            if ekey(n, m) in used:
                continue
            paths.append(walk(n, m))

    remaining = [ekey(p, q) for p in pts for q in neighbors[p] if ekey(p, q) not in used]
    for a, b in remaining:
        if ekey(a, b) in used:
            continue
        path = [a, b]
        used.add(ekey(a, b))
        prev, cur = a, b
        while True:
            nxt = [q for q in neighbors[cur] if q != prev and ekey(cur, q) not in used]
            if not nxt:
                break
            q = nxt[0]
            used.add(ekey(cur, q))
            path.append(q)
            prev, cur = cur, q
            if q == a:
                path.pop()
                path.append(a)
                break
        paths.append(path)

    return paths


def prune_and_simplify(paths: list, tolerance: float, scale: float) -> list[VectorPath]:
    result = []
    for path in paths:
        if len(path) < 2:
            continue
        ys = np.array([p[0] for p in path], dtype=np.float64)
        xs = np.array([p[1] for p in path], dtype=np.float64)
        length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        if length < 3.0 and len(path) < 5:
            continue
        coords = np.column_stack([xs, ys])
        closed = np.allclose(coords[0], coords[-1]) and len(coords) > 3
        if closed:
            coords = np.vstack([coords, coords[0]])
        simple = approximate_polygon(coords, tolerance=tolerance)
        if len(simple) < 2:
            continue
        simple = simple / scale
        pts = [(round(float(x), 2), round(float(y), 2)) for x, y in simple]
        if closed:
            pts[-1] = pts[0]
        result.append(VectorPath(points=pts, closed=closed))
    return result


def extract_outlines(mask: np.ndarray, tolerance: float, scale: float) -> list[VectorPath]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = []
    for c in contours:
        if cv2.contourArea(c) < 4.0 and cv2.arcLength(c, False) < 12.0:
            continue
        simple = approximate_polygon(c.reshape(-1, 2).astype(np.float64), tolerance=tolerance)
        if len(simple) < 3:
            continue
        simple = simple / scale
        pts = [(round(float(x), 2), round(float(y), 2)) for x, y in simple]
        result.append(VectorPath(points=pts, closed=True))
    return result


def write_svg(paths: list[VectorPath], width: int, height: int, stroke_width: float, out: Path) -> None:
    w = max(0.25, min(6.0, stroke_width))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <g fill="none" stroke="#000000" stroke-width="{w:.2f}" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for p in paths:
        d = f"M {p.points[0][0]} {p.points[0][1]}"
        for x, y in p.points[1:]:
            d += f" L {x} {y}"
        if p.closed:
            d += " Z"
        lines.append(f"    <path d=\"{d}\"/>")
    lines.append("  </g>")
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dxf(paths: list[VectorPath], height: int, unit: str, user_scale: float, out: Path) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = INSUNITS.get(unit, 0)
    doc.header["$MEASUREMENT"] = 1 if unit in ("mm", "cm") else 0
    doc.layers.add("TRACE", color=7)
    msp = doc.modelspace()
    k = user_scale
    for p in paths:
        pts = [(x * k, (height - y) * k) for x, y in p.points]
        msp.add_lwpolyline(pts, format="xy", close=p.closed, dxfattribs={"layer": "TRACE"})
    doc.saveas(out)


def write_preview(paths: list[VectorPath], width: int, height: int, out: Path) -> None:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for p in paths:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in p.points], dtype=np.int32)
        cv2.polylines(canvas, [pts], p.closed, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


class ImageVectorAgent:
    def __init__(self, mode: str = "auto", quality: str = "high", tolerance: float | None = None,
                 threshold: str = "auto", invert: bool = False, unit: str = "px", scale: float = 1.0,
                 min_area_frac: float = 2e-5):
        self.mode = mode
        self.quality = quality
        self.tolerance = tolerance
        self.threshold = threshold
        self.invert = invert
        self.unit = unit
        self.scale = scale
        self.min_area_frac = min_area_frac

    def run(self, src: Path, outdir: Path) -> dict:
        outdir.mkdir(parents=True, exist_ok=True)
        img = load_image(src)
        h0, w0 = img.shape[:2]
        gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray, scale = preprocess(gray0)
        mask, tmethod = binarize(gray, self.threshold, self.invert)
        mask = remove_specks(mask, self.min_area_frac)
        ink_fraction = float(np.count_nonzero(mask)) / mask.size
        if ink_fraction < 0.0005:
            raise ValueError("Aucun tracé détecté : image quasi vide ou paramètres inadaptés.")

        stroke_up = estimate_stroke_width(mask)
        if self.tolerance is not None:
            tol = self.tolerance * scale
        else:
            tol = QUALITY_TOLERANCE[self.quality] * scale

        mode = self.mode
        if mode == "auto":
            mode = "centerline" if ink_fraction < 0.35 else "outline"

        if mode == "centerline":
            skel = skeletonize(mask > 0).astype(np.uint8) * 255
            paths = prune_and_simplify(trace_skeleton(skel), tol, scale)
        else:
            paths = extract_outlines(mask, tol, scale)

        stroke_out = stroke_up / scale

        svg_path = outdir / (src.stem + ".svg")
        dxf_path = outdir / (src.stem + ".dxf")
        preview_path = outdir / (src.stem + "_preview.png")
        write_svg(paths, w0, h0, stroke_out, svg_path)
        write_dxf(paths, h0, self.unit, self.scale, dxf_path)
        write_preview(paths, w0, h0, preview_path)

        analysis = Analysis(
            width=w0, height=h0,
            contrast_std=round(float(gray0.std()), 2),
            blur_variance=round(float(cv2.Laplacian(gray0, cv2.CV_64F).var()), 2),
            noise_sigma=round(float(np.median(np.abs(cv2.Laplacian(gray0, cv2.CV_64F))) / 0.6745), 2),
            ink_fraction=round(ink_fraction, 4),
            ink_polarity="sombre" if not self.invert else "clair",
            threshold_method=tmethod,
            stroke_width_px=round(stroke_out, 2),
            mode_used=mode,
        )
        report = {
            "source": str(src),
            "analysis": asdict(analysis),
            "parameters": {"mode": self.mode, "quality": self.quality, "tolerance_px": round(tol / scale, 3),
                           "threshold": self.threshold, "invert": self.invert, "unit": self.unit, "scale": self.scale},
            "paths": len(paths),
            "points": sum(len(p.points) for p in paths),
            "outputs": {"svg": str(svg_path), "dxf": str(dxf_path), "preview": str(preview_path)},
        }
        (outdir / (src.stem + "_report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agent analyste d'image : JPEG -> SVG + DXF")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("sortie"))
    ap.add_argument("--mode", choices=["auto", "centerline", "outline"], default="auto")
    ap.add_argument("--quality", choices=["max", "high", "balanced"], default="high")
    ap.add_argument("--tolerance", type=float, default=None, help="Écart max de simplification en px image")
    ap.add_argument("--threshold", choices=["auto", "otsu", "adaptive"], default="auto")
    ap.add_argument("--invert", action="store_true", help="Tracé clair sur fond sombre")
    ap.add_argument("--unit", choices=["px", "mm", "cm", "in"], default="px")
    ap.add_argument("--scale", type=float, default=1.0, help="Facteur d'échelle des coordonnées DXF")
    ap.add_argument("--min-area", type=float, default=2e-5, help="Fraction de surface minimale des composantes gardées")
    args = ap.parse_args(argv)

    agent = ImageVectorAgent(mode=args.mode, quality=args.quality, tolerance=args.tolerance,
                             threshold=args.threshold, invert=args.invert, unit=args.unit, scale=args.scale,
                             min_area_frac=args.min_area)
    try:
        report = agent.run(args.image, args.outdir)
    except (ValueError, FileNotFoundError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
