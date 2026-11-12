#!/usr/bin/env python3
"""Agent analyste d'image : JPEG -> tracé vectoriel SVG + DXF.

Usage:
    python agent.py entree.jpg -o sortie/ [--mode auto|centerline|outline]
                    [--quality max|high|balanced] [--unit mm|cm|px] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

import ezdxf
from scipy.spatial import cKDTree

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
    stroke_width_mode: str
    mode_used: str


@dataclass
class VectorPath:
    points: list
    closed: bool
    kind: str = "poly"
    center: tuple | None = None
    radius: float = 0.0
    start_angle: float = 0.0
    sweep: float = 0.0


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


def extract_edges(gray: np.ndarray, tolerance: float, scale: float) -> list[VectorPath]:
    smooth = cv2.bilateralFilter(gray, 7, 40, 40)
    med = float(np.median(smooth))
    lo = int(max(15.0, min(150.0, med * 0.60)))
    hi = int(max(45.0, min(230.0, med * 1.20)))
    edges = cv2.Canny(smooth, lo, hi)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    skel = skeletonize(edges > 0).astype(np.uint8) * 255
    return prune_and_simplify(trace_skeleton(skel), tolerance, scale)


def extract_outlines(mask: np.ndarray, tolerance: float, scale: float) -> list[VectorPath]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    min_area = max(8.0, 5e-5 * h * w)
    min_perimeter = max(24.0, 0.03 * max(h, w))
    result = []
    for c in contours:
        if cv2.contourArea(c) < min_area and cv2.arcLength(c, False) < min_perimeter:
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
        if p.kind == "circle":
            cx, cy = p.center
            lines.append(f'    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{p.radius:.2f}"/>')
            continue
        if p.kind == "arc":
            pts = curve_points(p)
            (sx, sy), (ex, ey) = pts[0], pts[-1]
            large = 1 if abs(p.sweep) > 180 else 0
            sweep_flag = 1 if p.sweep > 0 else 0
            lines.append(f'    <path d="M {sx:.2f} {sy:.2f} A {p.radius:.2f} {p.radius:.2f} 0 {large} {sweep_flag} {ex:.2f} {ey:.2f}"/>')
            continue
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
        if p.kind == "circle":
            cx, cy = p.center
            msp.add_circle((cx * k, (height - cy) * k), p.radius * k, dxfattribs={"layer": "TRACE"})
            continue
        if p.kind == "arc":
            cx, cy = p.center
            a0 = (-p.start_angle) % 360.0
            sweep_d = -p.sweep
            if sweep_d >= 0:
                start, end = a0, (a0 + sweep_d) % 360.0
            else:
                start, end = (a0 + sweep_d) % 360.0, a0
            msp.add_arc((cx * k, (height - cy) * k), p.radius * k, start, end, dxfattribs={"layer": "TRACE"})
            continue
        pts = [(x * k, (height - y) * k) for x, y in p.points]
        msp.add_lwpolyline(pts, format="xy", close=p.closed, dxfattribs={"layer": "TRACE"})
    doc.saveas(out)


def write_preview(paths: list[VectorPath], width: int, height: int, out: Path) -> None:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for p in paths:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in curve_points(p)], dtype=np.int32)
        cv2.polylines(canvas, [pts], p.closed or p.kind == "circle", (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def circle_fit(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    Amat = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    bvec = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    err = np.abs(np.hypot(x - cx, y - cy) - r)
    return float(cx), float(cy), r, float(err.max())


def detect_curves(paths: list[VectorPath], tol: float, max_dim: float) -> list[VectorPath]:
    result = []
    for p in paths:
        if p.kind != "poly" or len(p.points) < 8:
            result.append(p)
            continue
        pts = np.array(p.points, dtype=np.float64)
        fit = circle_fit(pts)
        if fit is None:
            result.append(p)
            continue
        cx, cy, r, emax = fit
        if r < 4.0 or r > 2.0 * max_dim or emax > max(1.5 * tol, 0.03 * r):
            result.append(p)
            continue
        ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        unwrapped = np.unwrap(np.radians(ang))
        sweep = float(np.degrees(unwrapped[-1] - unwrapped[0]))
        if float(np.abs(np.diff(unwrapped)).max()) > math.radians(45.0):
            result.append(p)
            continue
        if abs(sweep) >= 320.0:
            result.append(VectorPath(points=[], closed=True, kind="circle", center=(cx, cy), radius=r))
        elif 25.0 <= abs(sweep) <= 330.0:
            result.append(VectorPath(points=[], closed=False, kind="arc", center=(cx, cy), radius=r,
                                     start_angle=float(ang[0]) % 360.0, sweep=sweep))
        else:
            result.append(p)
    return result


def edge_coverage_map(gray: np.ndarray):
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(smooth, 10, 30)
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    gblur = cv2.GaussianBlur(gray, (9, 9), 2.0).astype(np.float32)
    return dist, gblur


def ring_scores(dist: np.ndarray, gblur: np.ndarray, cx: float, cy: float, r: float):
    """Couverture d'arêtes, prominence radiale et contraste de luminance d'un cercle."""
    H, W = dist.shape
    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    def coverage(rr: float) -> float:
        px = np.clip(np.round(cx + rr * ct).astype(int), 0, W - 1)
        py = np.clip(np.round(cy + rr * st).astype(int), 0, H - 1)
        return float((dist[py, px] <= 3.0).mean())

    cov = coverage(r)
    inner = min(coverage(r - d) for d in range(6, 26, 2))
    outer = min(coverage(r + d) for d in range(6, 26, 2))
    step = 0.0
    for d in (3, 5, 8):
        pi = np.clip(np.round(cx + (r - d) * ct).astype(int), 0, W - 1)
        qi = np.clip(np.round(cy + (r - d) * st).astype(int), 0, H - 1)
        po = np.clip(np.round(cx + (r + d) * ct).astype(int), 0, W - 1)
        qo = np.clip(np.round(cy + (r + d) * st).astype(int), 0, H - 1)
        step = max(step, float(np.abs(gblur[qi, pi] - gblur[qo, po]).mean()))
    return cov, cov - max(inner, outer), step


def detect_rings(gray: np.ndarray, mask: np.ndarray, scale: float,
                 emap=None) -> list[VectorPath]:
    """Détecte les anneaux circulaires (grands cercles concentriques) par couverture d'arêtes.

    Les anneaux gravés/imprimés ne sont pas toujours des frontières encre/fond du
    masque binaire : on cherche donc des cercles le long desquels les arêtes (Canny
    à seuils bas) sont présentes et isolées radialement, autour d'un centre dominant
    déduit des meilleurs candidats géométriques du masque.
    """
    H, W = gray.shape
    max_dim = float(max(H, W))
    if emap is None:
        emap = edge_coverage_map(gray)
    dist, gblur = emap
    if np.count_nonzero(dist == 0) < 50:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cands = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 40:
            continue
        fit = circle_fit(pts[::2])
        if fit is None:
            continue
        cx, cy, r, emax = fit
        if not (0.04 * max_dim <= r <= 0.8 * max_dim) or emax > max(2.0, 0.02 * r):
            continue
        cands.append([cx, cy, r, float(len(pts))])

    if cands:
        cluster_tol = 0.08 * max_dim
        used = [False] * len(cands)
        clusters = []
        for i in range(len(cands)):
            if used[i]:
                continue
            members = [i]
            used[i] = True
            for j in range(i + 1, len(cands)):
                if not used[j] and math.hypot(cands[i][0] - cands[j][0], cands[i][1] - cands[j][1]) < cluster_tol:
                    members.append(j)
                    used[j] = True
            wsum = sum(cands[m][3] for m in members)
            cx = sum(cands[m][0] * cands[m][3] for m in members) / wsum
            cy = sum(cands[m][1] * cands[m][3] for m in members) / wsum
            clusters.append((cx, cy, wsum))
        clusters.sort(key=lambda t: -t[2])
        center = (clusters[0][0], clusters[0][1])
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        center = (float(xs.mean()), float(ys.mean()))

    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    def coverage(cx: float, cy: float, r: float) -> float:
        px = np.clip(np.round(cx + r * ct).astype(int), 0, W - 1)
        py = np.clip(np.round(cy + r * st).astype(int), 0, H - 1)
        return float((dist[py, px] <= 3.0).mean())

    def prominence(cx: float, cy: float, r: float) -> float:
        c0 = coverage(cx, cy, r)
        inner = min(coverage(cx, cy, r - d) for d in range(6, 26, 2))
        outer = min(coverage(cx, cy, r + d) for d in range(6, 26, 2))
        return c0 - max(inner, outer)

    def step_score(cx: float, cy: float, r: float) -> float:
        score = 0.0
        for d in (3, 5, 8):
            pi = np.clip(np.round(cx + (r - d) * ct).astype(int), 0, W - 1)
            qi = np.clip(np.round(cy + (r - d) * st).astype(int), 0, H - 1)
            po = np.clip(np.round(cx + (r + d) * ct).astype(int), 0, W - 1)
            qo = np.clip(np.round(cy + (r + d) * st).astype(int), 0, H - 1)
            score = max(score, float(np.abs(gblur[qi, pi] - gblur[qo, po]).mean()))
        return score

    def refine(cx: float, cy: float, r: float):
        cov = coverage(cx, cy, r)
        for step in (4.0, 2.0, 1.0):
            improved = True
            while improved:
                improved = False
                for dx, dy, dr in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                                   (0, 0, step), (0, 0, -step)):
                    c2 = coverage(cx + dx, cy + dy, r + dr)
                    if c2 > cov + 1e-6:
                        cx, cy, r, cov = cx + dx, cy + dy, r + dr, c2
                        improved = True
        return cx, cy, r, cov

    radii = np.arange(max(8.0, 0.04 * max_dim), 0.85 * max_dim, 2.0)
    if len(radii) < 3:
        return []
    covs = np.array([coverage(center[0], center[1], r) for r in radii])
    peaks = [float(radii[i]) for i in range(1, len(radii) - 1)
             if covs[i] >= 0.5 and covs[i] >= covs[i - 1] and covs[i] >= covs[i + 1]]

    kept = []
    for r0 in peaks:
        cx, cy, r, cov = refine(center[0], center[1], r0)
        cov2, prom, step = ring_scores(dist, gblur, cx, cy, r)
        if cov2 < 0.7 or prom < 0.2 or step < 30.0:
            continue
        if not (0.04 * max_dim <= r <= 0.8 * max_dim):
            continue
        if math.hypot(cx - center[0], cy - center[1]) > 0.06 * max_dim:
            continue
        if any(math.hypot(cx - kx, cy - ky) < 10 and abs(r - kr) < max(4.0, 0.02 * r)
               for kx, ky, kr, _ in kept):
            continue
        kept.append((cx, cy, r, cov))
    kept.sort(key=lambda t: -t[2])
    return [VectorPath(points=[], closed=True, kind="circle",
                       center=(cx / scale, cy / scale), radius=r / scale)
            for cx, cy, r, _ in kept]


def strip_points_near_circles(paths: list[VectorPath], circles: list[VectorPath], band: float) -> list[VectorPath]:
    if not circles:
        return paths
    out = []
    for p in paths:
        if p.kind != "poly" or len(p.points) < 2:
            out.append(p)
            continue
        pts = np.array(p.points, dtype=np.float64)
        keep = np.ones(len(pts), dtype=bool)
        for c in circles:
            d = np.hypot(pts[:, 0] - c.center[0], pts[:, 1] - c.center[1])
            keep &= np.abs(d - c.radius) > band
        if keep.all():
            out.append(p)
            continue
        idx = np.nonzero(keep)[0]
        if len(idx) < 2:
            continue
        segs = []
        cur = [int(idx[0])]
        for i in idx[1:]:
            if i == cur[-1] + 1:
                cur.append(int(i))
            else:
                segs.append(cur)
                cur = [int(i)]
        segs.append(cur)
        if p.closed and len(segs) > 1 and 0 in segs[0] and (len(pts) - 1) in segs[-1]:
            segs[0] = segs[-1] + segs[0]
            segs.pop()
        for seg in segs:
            seg_pts = [(round(float(pts[i][0]), 2), round(float(pts[i][1]), 2)) for i in seg]
            if len(seg_pts) >= 2:
                out.append(VectorPath(points=seg_pts, closed=False))
    return out


def curve_points(p: VectorPath, step_deg: float = 5.0) -> list:
    if p.kind == "circle":
        n = max(24, int(360 / step_deg))
        return [(p.center[0] + p.radius * math.cos(math.radians(360 * i / n)),
                 p.center[1] + p.radius * math.sin(math.radians(360 * i / n))) for i in range(n)]
    if p.kind == "arc":
        n = max(4, int(abs(p.sweep) / step_deg))
        return [(p.center[0] + p.radius * math.cos(math.radians(p.start_angle + p.sweep * i / n)),
                 p.center[1] + p.radius * math.sin(math.radians(p.start_angle + p.sweep * i / n))) for i in range(n + 1)]
    return p.points


def close_paths(paths: list[VectorPath], gap: float, closed_only: bool) -> list[VectorPath]:
    result = [p for p in paths if p.closed or p.kind != "poly"]
    opens = [p for p in paths if not p.closed and p.kind == "poly"]
    if not opens:
        return result

    ends = []
    for i, p in enumerate(opens):
        ends.append((i, 0, p.points[0]))
        ends.append((i, 1, p.points[-1]))
    coords = np.array([[e[2][0], e[2][1]] for e in ends])
    max_dim = float(coords[:, 0].max() - coords[:, 0].min() + coords[:, 1].max() - coords[:, 1].min()) / 2.0

    chains = {}
    for i, p in enumerate(opens):
        chains[i] = {"pts": list(p.points), "e_start": (i, 0), "e_end": (i, 1), "closed": False}
    end_chain = {}
    for i in chains:
        end_chain[(i, 0)] = i
        end_chain[(i, 1)] = i

    def oriented(chain, end_id):
        if chain["e_start"] == end_id:
            return chain["pts"], chain["e_end"]
        return chain["pts"][::-1], chain["e_start"]

    def match_pass(tol: float) -> None:
        free = [eid for eid, cid in end_chain.items() if not chains[cid]["closed"]]
        if len(free) < 2:
            return
        idx_of = {eid: k for k, eid in enumerate(free)}
        pts = np.array([ends[2 * eid[0] + eid[1]][2] for eid in free])
        tree = cKDTree(pts)
        raw = tree.query_pairs(tol, output_type="ndarray")
        degree = np.zeros(len(free), dtype=int)
        candidates = []
        for a, b in raw:
            a, b = int(a), int(b)
            ea, eb = free[a], free[b]
            if ea[0] == eb[0]:
                continue
            d = float(np.hypot(*(pts[a] - pts[b])))
            candidates.append((d, ea, eb))
            degree[a] += 1
            degree[b] += 1
        candidates.sort()
        phase1 = [c for c in candidates if degree[idx_of[c[1]]] == 1 and degree[idx_of[c[2]]] == 1]
        phase2 = [c for c in candidates if not (degree[idx_of[c[1]]] == 1 and degree[idx_of[c[2]]] == 1)]

        def opposite_distance(ea, eb):
            oa = 2 * ea[0] + (1 - ea[1])
            ob = 2 * eb[0] + (1 - eb[1])
            return float(np.hypot(*(coords[oa] - coords[ob])))

        phase2.sort(key=lambda c: (-opposite_distance(c[1], c[2]), c[0]))

        for _, ea, eb in phase1 + phase2:
            ca = end_chain.get(ea)
            cb = end_chain.get(eb)
            if ca is None or cb is None:
                continue
            if ca == cb:
                C = chains[ca]
                if not C["closed"]:
                    C["pts"].append(C["pts"][0])
                    C["closed"] = True
                    end_chain.pop(C["e_start"], None)
                    end_chain.pop(C["e_end"], None)
                    C["e_start"] = C["e_end"] = None
                continue
            A, B = chains[ca], chains[cb]
            if A["closed"] or B["closed"]:
                continue
            pa, other_a = oriented(A, ea)
            pb, other_b = oriented(B, eb)
            merged = {"pts": pa[::-1] + pb, "e_start": other_a, "e_end": other_b, "closed": False}
            del chains[ca], chains[cb]
            end_chain.pop(ea, None)
            end_chain.pop(eb, None)
            new_id = max(chains, default=-1) + 1
            chains[new_id] = merged
            if merged["e_start"] is not None:
                end_chain[merged["e_start"]] = new_id
                end_chain[merged["e_end"]] = new_id

    for factor in (1.0, 2.0, 4.0):
        match_pass(gap * factor)

    def self_bridge(cid) -> None:
        c = chains[cid]
        c["pts"].append(c["pts"][0])
        c["closed"] = True
        c["e_start"] = c["e_end"] = None

    bridge_limit = 0.10 * max_dim
    for cid in list(chains):
        c = chains[cid]
        if not c["closed"] and math.dist(c["pts"][0], c["pts"][-1]) <= bridge_limit:
            self_bridge(cid)

    far_ids = [cid for cid, c in chains.items() if not c["closed"]]
    while len(far_ids) > 1:
        best = None
        for ii in range(len(far_ids)):
            for jj in range(ii + 1, len(far_ids)):
                A, B = chains[far_ids[ii]], chains[far_ids[jj]]
                for ea in (A["e_start"], A["e_end"]):
                    for eb in (B["e_start"], B["e_end"]):
                        d = math.dist(ends[2 * ea[0] + ea[1]][2], ends[2 * eb[0] + eb[1]][2])
                        if best is None or d < best[0]:
                            best = (d, ii, jj, ea, eb)
        if best[0] > bridge_limit:
            break
        _, ii, jj, ea, eb = best
        cid_a, cid_b = far_ids[ii], far_ids[jj]
        A, B = chains[cid_a], chains[cid_b]
        pa, other_a = oriented(A, ea)
        pb, other_b = oriented(B, eb)
        merged = {"pts": pa[::-1] + pb, "e_start": other_a, "e_end": other_b, "closed": False}
        del chains[cid_a], chains[cid_b]
        end_chain.pop(ea, None)
        end_chain.pop(eb, None)
        new_id = max(chains, default=-1) + 1
        chains[new_id] = merged
        far_ids = [x for k, x in enumerate(far_ids) if k not in (ii, jj)]
        if not merged["closed"]:
            far_ids.append(new_id)
    for cid in list(chains):
        c = chains[cid]
        if not c["closed"] and math.dist(c["pts"][0], c["pts"][-1]) <= bridge_limit:
            self_bridge(cid)

    for c in chains.values():
        if c["closed"] or not closed_only:
            result.append(VectorPath(points=c["pts"], closed=c["closed"]))
    return result


class ImageVectorAgent:
    def __init__(self, mode: str = "auto", quality: str = "high", tolerance: float | None = None,
                 threshold: str = "auto", invert: bool = False, unit: str = "px", scale: float = 1.0,
                 min_area_frac: float = 2e-5, close_lines: bool = False, closed_only: bool = False,
                 gap: float | None = None, stroke_width: float | None = None,
                 fit_curves: bool = True):
        self.mode = mode
        self.quality = quality
        self.tolerance = tolerance
        self.threshold = threshold
        self.invert = invert
        self.unit = unit
        self.scale = scale
        self.min_area_frac = min_area_frac
        self.close_lines = close_lines
        self.closed_only = closed_only
        self.gap = gap
        self.stroke_width = stroke_width
        self.fit_curves = fit_curves

    def run(self, src: Path, outdir: Path) -> dict:
        outdir.mkdir(parents=True, exist_ok=True)
        img = load_image(src)
        h0, w0 = img.shape[:2]
        gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray, scale = preprocess(gray0)
        invert = self.invert
        polarity = "clair" if invert else "sombre"
        if not invert:
            b = 4
            border = np.concatenate([
                gray0[:b, :].ravel(), gray0[-b:, :].ravel(),
                gray0[:, :b].ravel(), gray0[:, -b:].ravel(),
            ])
            if float(border.mean()) < 110.0:
                invert = True
                polarity = "clair"
        mask, tmethod = binarize(gray, self.threshold, invert)
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
            if len(paths) > 1.5 * max(w0, h0):
                mode = "edges"
                paths = extract_edges(gray, tol, scale)
        elif mode == "edges":
            paths = extract_edges(gray, tol, scale)
        else:
            paths = extract_outlines(mask, tol, scale)

        closed_count = sum(1 for p in paths if p.closed)
        if self.close_lines:
            gap = self.gap if self.gap is not None else max(1.5, 0.004 * max(w0, h0))
            paths = close_paths(paths, gap, self.closed_only)
            closed_count = sum(1 for p in paths if p.closed)
        if self.fit_curves:
            paths = detect_curves(paths, tol / scale, max(w0, h0))
            emap = edge_coverage_map(gray)
            verified = []
            for p in paths:
                if p.kind == "circle":
                    cov, prom, step = ring_scores(emap[0], emap[1],
                                                  p.center[0] * scale, p.center[1] * scale,
                                                  p.radius * scale)
                    if cov < 0.7 or step < 30.0:
                        p = VectorPath(points=curve_points(p), closed=True)
                verified.append(p)
            paths = verified
            ring_circles = detect_rings(gray, mask, scale, emap=emap)
            existing = [p for p in paths if p.kind == "circle"
                        or (p.kind == "arc" and abs(p.sweep) >= 180.0)]
            new_circles = []
            for c in ring_circles:
                if any(math.hypot(c.center[0] - e.center[0], c.center[1] - e.center[1]) < 10 and
                       abs(c.radius - e.radius) < max(stroke_up / scale, 4.0) for e in existing):
                    continue
                if any(math.hypot(c.center[0] - n.center[0], c.center[1] - n.center[1]) < 8 and
                       abs(c.radius - n.radius) < max(3.0, 0.02 * n.radius) for n in new_circles):
                    continue
                new_circles.append(c)
            if new_circles:
                all_circles = [p for p in paths if p.kind == "circle"] + new_circles
                band = max(2.5, 0.75 * stroke_up / scale)
                paths = strip_points_near_circles(paths, all_circles, band) + new_circles
        n_circles = sum(1 for p in paths if p.kind == "circle")
        n_arcs = sum(1 for p in paths if p.kind == "arc")

        stroke_out = self.stroke_width if self.stroke_width is not None else stroke_up / scale

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
            ink_polarity=polarity,
            threshold_method=tmethod,
            stroke_width_px=round(stroke_out, 2),
            stroke_width_mode="manuel" if self.stroke_width is not None else "auto",
            mode_used=mode,
        )
        report = {
            "source": str(src),
            "analysis": asdict(analysis),
            "parameters": {"mode": self.mode, "mode_effectif": mode, "quality": self.quality,
                           "tolerance_px": round(tol / scale, 3),
                           "threshold": self.threshold, "invert": invert, "unit": self.unit, "scale": self.scale,
                           "close_lines": self.close_lines, "closed_only": self.closed_only, "gap_px": self.gap},
            "paths": len(paths),
            "closed_paths": closed_count,
            "circles": n_circles,
            "arcs": n_arcs,
            "points": sum(len(p.points) for p in paths),
            "outputs": {"svg": str(svg_path), "dxf": str(dxf_path), "preview": str(preview_path),
                         "report": str(outdir / (src.stem + "_report.json"))},
        }
        (outdir / (src.stem + "_report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agent analyste d'image : JPEG -> SVG + DXF")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("sortie"))
    ap.add_argument("--mode", choices=["auto", "centerline", "outline", "edges"], default="auto")
    ap.add_argument("--quality", choices=["max", "high", "balanced"], default="high")
    ap.add_argument("--tolerance", type=float, default=None, help="Écart max de simplification en px image")
    ap.add_argument("--threshold", choices=["auto", "otsu", "adaptive"], default="auto")
    ap.add_argument("--invert", action="store_true", help="Tracé clair sur fond sombre")
    ap.add_argument("--unit", choices=["px", "mm", "cm", "in"], default="px")
    ap.add_argument("--scale", type=float, default=1.0, help="Facteur d'échelle des coordonnées DXF")
    ap.add_argument("--min-area", type=float, default=2e-5, help="Fraction de surface minimale des composantes gardées")
    ap.add_argument("--close", action="store_true", help="Fermer les tracés : raccord des extrémités proches (gravure laser)")
    ap.add_argument("--closed-only", action="store_true", help="Ne conserver que les tracés fermés")
    ap.add_argument("--gap", type=float, default=None, help="Distance max de raccord des extrémités en px (défaut auto)")
    ap.add_argument("--stroke-width", type=float, default=None, help="Épaisseur du trait SVG en px (défaut : estimation auto)")
    ap.add_argument("--no-curves", action="store_true", help="Désactiver la détection de cercles/arcs parfaits")
    args = ap.parse_args(argv)

    agent = ImageVectorAgent(mode=args.mode, quality=args.quality, tolerance=args.tolerance,
                             threshold=args.threshold, invert=args.invert, unit=args.unit, scale=args.scale,
                             min_area_frac=args.min_area, close_lines=args.close,
                             closed_only=args.closed_only, gap=args.gap, stroke_width=args.stroke_width,
                             fit_curves=not args.no_curves)
    try:
        report = agent.run(args.image, args.outdir)
    except (ValueError, FileNotFoundError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
