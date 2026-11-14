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
    axis_a: float = 0.0
    axis_b: float = 0.0
    tilt: float = 0.0


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
        if p.kind == "ellipse":
            cx, cy = p.center
            lines.append(f'    <ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{p.axis_a:.2f}" ry="{p.axis_b:.2f}" '
                         f'transform="rotate({p.tilt:.2f} {cx:.2f} {cy:.2f})"/>')
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
        if p.kind == "ellipse":
            cx, cy = p.center
            tr = math.radians(p.tilt)
            major = (p.axis_a * math.cos(tr) * k, -p.axis_a * math.sin(tr) * k)
            ratio = p.axis_b / p.axis_a if p.axis_a > 0 else 1.0
            msp.add_ellipse((cx * k, (height - cy) * k), major, ratio, dxfattribs={"layer": "TRACE"})
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


def build_ring_maps(gray: np.ndarray):
    """Cartes d'arêtes pour la détection d'anneaux.

    - dist : distance de chaque pixel à l'arête (Canny seuils bas) la plus proche ;
    - labels/ang/gmag : pour chaque pixel, l'arête la plus proche (identifiant),
      l'orientation de son gradient et son amplitude ;
    - gblur : luminance lissée pour mesurer le contraste de part et d'autre d'un cercle.
    """
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(smooth, 10, 30)
    dist, labels = cv2.distanceTransformWithLabels((edges == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    eys, exs = np.nonzero(edges)
    lab_ids = labels[eys, exs]
    ang = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    ang[lab_ids] = np.arctan2(gy[eys, exs], gx[eys, exs])
    gmag = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    gmag[lab_ids] = np.hypot(gx[eys, exs], gy[eys, exs])
    gblur = cv2.GaussianBlur(gray, (9, 9), 2.0).astype(np.float32)
    return dist, labels, ang, gmag, gblur


def ring_scores(maps, cx: float, cy: float, r: float):
    """Vérification d'un cercle : couvertures (simple et cohérente), plus grand
    secteur occlus, contraste de luminance. Coords image up-scalées."""
    dist, labels, ang, gmag, gblur = maps
    H, W = dist.shape
    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    px = np.clip(np.round(cx + r * ct).astype(int), 0, W - 1)
    py = np.clip(np.round(cy + r * st).astype(int), 0, H - 1)
    near = dist[py, px] <= 3.0
    ga = ang[labels[py, px]]
    radial = np.arctan2(py - cy, px - cx)
    d = np.abs((ga - radial + np.pi) % (2.0 * np.pi) - np.pi)
    coh = near & (d < math.radians(45.0)) & (gmag[labels[py, px]] > 15.0)
    idx = np.nonzero(~near)[0]
    if len(idx) == 0:
        gap = 0.0
    else:
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        gap = max(len(x) for x in runs) / len(near)
        if not near[0] and not near[-1]:
            gap = max(gap, (len(runs[0]) + len(runs[-1])) / len(near))
    step = 0.0
    for dd in (3, 5, 8):
        if dd >= r * 0.8:
            continue
        pi = np.clip(np.round(cx + (r - dd) * ct).astype(int), 0, W - 1)
        qi = np.clip(np.round(cy + (r - dd) * st).astype(int), 0, H - 1)
        po = np.clip(np.round(cx + (r + dd) * ct).astype(int), 0, W - 1)
        qo = np.clip(np.round(cy + (r + dd) * st).astype(int), 0, H - 1)
        step = max(step, float(np.abs(gblur[qi, pi] - gblur[qo, po]).mean()))
    return float(near.mean()), float(coh.mean()), float(gap), float(step)


def detect_rings(gray: np.ndarray, mask: np.ndarray, scale: float,
                 maps=None) -> list[VectorPath]:
    """Détecte les cercles parfaits, y compris partiels (occlus), non concentriques
    et à double piste.

    Candidats : balayage radial autour des 4 principaux centres de clusters de
    contours (anneaux concentriques multi-centres) et transformée de Hough
    (petits cercles multi-centres, sans plafond arbitraire : top 1000 triés
    par couverture). Chaque candidat est affiné par hill-climbing sur la
    couverture d'arêtes (240 angles), puis validé par : couverture stricte
    (>= 0.7, occlusion <= 30 %) ou cohérente (gradient radial >= 0.55),
    proéminence radiale avec fenêtre dépendant de la taille (seuil 0.12 pour
    les grands cercles, 0.15 sinon), contraste de luminance, et dispersion
    radiale (std/mean du rayon réel des arêtes <= 0.05) qui rejette les
    glyphes ressemblant à des cercles (lettres O, etc.).

    Limites connues sur photos très texturées : cercles internes des
    engrenages à double piste serrée et petits cercles noyés dans la texture
    (rivets) peuvent être rejetés par la proéminence ; ils restent alors
    présents comme tracés généraux (polylignes) dans le SVG/DXF.
    """
    dist, labels, ang, gmag, gblur = maps if maps is not None else build_ring_maps(gray)
    H, W = dist.shape
    max_dim = float(max(H, W))
    if np.count_nonzero(dist == 0) < 50:
        return []

    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    thr = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    thq = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    thp = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    crt, srt = np.cos(thr), np.sin(thr)
    cqt, sqt = np.cos(thq), np.sin(thq)
    cpt, spt = np.cos(thp), np.sin(thp)

    def sample(cx, cy, r, cos_t, sin_t):
        px = np.clip(np.round(cx + r * cos_t).astype(int), 0, W - 1)
        py = np.clip(np.round(cy + r * sin_t).astype(int), 0, H - 1)
        return px, py

    def cov_at(cx, cy, r, cos_t, sin_t):
        px, py = sample(cx, cy, r, cos_t, sin_t)
        return float((dist[py, px] <= 3.0).mean())

    def near_hits(cx, cy, r, cos_t=None, sin_t=None):
        cos_t = ct if cos_t is None else cos_t
        sin_t = st if sin_t is None else sin_t
        px, py = sample(cx, cy, r, cos_t, sin_t)
        return dist[py, px] <= 3.0

    def quick_cov(cx, cy, r):
        px, py = sample(cx, cy, r, cqt, sqt)
        return float((dist[py, px] <= 3.0).mean())

    def cov_fast(cx, cy, r):
        px, py = sample(cx, cy, r, crt, srt)
        return float((dist[py, px] <= 3.0).mean())

    def coh_hits(cx, cy, r, cos_t=None, sin_t=None):
        cos_t = ct if cos_t is None else cos_t
        sin_t = st if sin_t is None else sin_t
        px, py = sample(cx, cy, r, cos_t, sin_t)
        near = dist[py, px] <= 3.0
        lab = labels[py, px]
        d = np.abs((ang[lab] - np.arctan2(py - cy, px - cx) + np.pi) % (2.0 * np.pi) - np.pi)
        return near & (d < math.radians(45.0)) & (gmag[lab] > 15.0)

    def gap_of(hit):
        idx = np.nonzero(~hit)[0]
        n = len(hit)
        if len(idx) == 0:
            return 0.0
        if len(idx) == n:
            return 1.0
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        g = max(len(x) for x in runs) / n
        if not hit[0] and not hit[-1]:
            g = max(g, (len(runs[0]) + len(runs[-1])) / n)
        return g

    def prom_of(hitfn, cx, cy, r, center_cov, cos_t, sin_t):
        wf = 0.12 if r < 0.15 * max_dim else 0.045
        d0, d1 = max(4.0, 0.03 * r), max(10.0, wf * r)
        offs = np.arange(d0, d1, max(2.0, (d1 - d0) / 7.0))
        sides = [float(hitfn(cx, cy, r + s * d, cos_t, sin_t).mean())
                 for d in offs for s in (-1.0, 1.0)]
        return center_cov - max(sides)

    def dispersion(cx, cy, r):
        n = int(min(1440, max(360, math.pi * r)))
        thd = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        drs = np.arange(-4.0, 4.5, 1.0)
        ct_d = np.cos(thd)
        st_d = np.sin(thd)
        stack = np.stack([
            dist[np.clip(np.round(cy + (r + dr) * st_d).astype(int), 0, H - 1),
                 np.clip(np.round(cx + (r + dr) * ct_d).astype(int), 0, W - 1)]
            for dr in drs])
        best = np.argmin(stack, axis=0)
        dmin = stack[best, np.arange(n)]
        sel = dmin <= 3.0
        if int(sel.sum()) < 30:
            return 1.0
        rr = r + drs[best[sel]]
        return float(rr.std() / rr.mean())

    def step_of(cx, cy, r, cos_t, sin_t):
        best = 0.0
        for d in (3, 5, 8):
            if d >= r * 0.8:
                continue
            pi, qi = sample(cx, cy, r - d, cos_t, sin_t)
            po, qo = sample(cx, cy, r + d, cos_t, sin_t)
            best = max(best, float(np.abs(gblur[qi, pi] - gblur[qo, po]).mean()))
        return best

    def refine(cx, cy, r):
        px, py = sample(cx, cy, r, crt, srt)
        cov = float((dist[py, px] <= 3.0).mean())
        moves = 0
        for step in (4.0, 2.0, 1.0):
            improved = True
            while improved and moves < 80:
                improved = False
                for dx, dy, dr in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                                   (0, 0, step), (0, 0, -step)):
                    qx, qy = sample(cx + dx, cy + dy, r + dr, crt, srt)
                    c2 = float((dist[qy, qx] <= 3.0).mean())
                    if c2 > cov + 1e-6:
                        cx, cy, r, cov = cx + dx, cy + dy, r + dr, c2
                        improved = True
                        moves += 1
        return cx, cy, r

    fits = []
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 40:
            continue
        fit = circle_fit(pts[::2])
        if fit is None:
            continue
        fx, fy, fr, femax = fit
        if not (0.04 * max_dim <= fr <= 0.8 * max_dim) or femax > max(2.0, 0.02 * fr):
            continue
        fits.append([fx, fy, fr, float(len(pts))])
    if fits:
        tol = 0.08 * max_dim
        used = [False] * len(fits)
        clusters = []
        for i in range(len(fits)):
            if used[i]:
                continue
            mem = [i]
            used[i] = True
            for j in range(i + 1, len(fits)):
                if not used[j] and math.hypot(fits[i][0] - fits[j][0], fits[i][1] - fits[j][1]) < tol:
                    mem.append(j)
                    used[j] = True
            w = sum(fits[m][3] for m in mem)
            clusters.append((sum(fits[m][0] * fits[m][3] for m in mem) / w,
                             sum(fits[m][1] * fits[m][3] for m in mem) / w, w))
        clusters.sort(key=lambda t: -t[2])
        centers = [(c[0], c[1]) for c in clusters[:4]]
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        centers = [(float(xs.mean()), float(ys.mean()))]

    cands = []
    for ccx, ccy in centers:
        for r in np.arange(max(8.0, 0.04 * max_dim), 0.85 * max_dim, 2.0):
            if cov_at(ccx, ccy, float(r), ct, st) >= 0.5:
                cands.append((ccx, ccy, float(r), ccx, ccy))

    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    hc = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=2.0, minDist=14, param1=60, param2=30,
                          minRadius=int(0.012 * max_dim), maxRadius=int(0.45 * max_dim))
    if hc is not None:
        for hx, hy, hr in hc[0]:
            hx, hy, hr = float(hx), float(hy), float(hr)
            if quick_cov(hx, hy, hr) < 0.55:
                continue
            cands.append((hx, hy, hr, None, None))

    cell = 24.0
    grid: dict[tuple[int, int], list] = {}
    pruned = []
    for cx, cy, r, scx, scy in cands:
        gx, gy = int(cx // cell), int(cy // cell)
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py, pr in grid.get((gx + dx, gy + dy), ()):
                    if math.hypot(cx - px, cy - py) < 6.0 and abs(r - pr) < max(4.0, 0.04 * r):
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if dup:
            continue
        pruned.append((cx, cy, r, scx, scy))
        grid.setdefault((gx, gy), []).append((cx, cy, r))

    radial = [c for c in pruned if c[3] is not None]
    hough = [c for c in pruned if c[3] is None]
    hough.sort(key=lambda c: -float(near_hits(c[0], c[1], c[2]).mean()))
    pruned = radial + hough[:1000]

    results = []
    for cx0, cy0, r0, scx, scy in pruned:
        cx, cy, r = refine(cx0, cy0, r0)
        if not (0.012 * max_dim <= r <= 0.8 * max_dim):
            continue
        if scx is not None and math.hypot(cx - scx, cy - scy) > 0.06 * max_dim:
            continue
        nh = near_hits(cx, cy, r)
        cov_n = float(nh.mean())
        gap_n = gap_of(nh)
        ch = coh_hits(cx, cy, r)
        cov_c = float(ch.mean())
        gap_c = gap_of(ch)
        alive_strict = cov_n >= 0.7 and gap_n <= 0.30
        alive_coh = cov_c >= 0.55 and gap_c <= 0.30
        if not (alive_strict or alive_coh):
            continue
        prom_n = prom_of(near_hits, cx, cy, r, float(cov_n), ct, st) if alive_strict else -1.0
        prom_c = prom_of(coh_hits, cx, cy, r, float(cov_c), ct, st) if alive_coh else -1.0
        pmin = 0.12 if r >= 0.15 * max_dim else 0.15
        if not ((alive_strict and prom_n >= pmin) or (alive_coh and prom_c >= pmin)):
            continue
        step = step_of(cx, cy, r, cpt, spt)
        ok_strict = alive_strict and prom_n >= pmin and step >= 30.0
        ok_coh = alive_coh and prom_c >= pmin and step >= 30.0
        if not (ok_strict or ok_coh):
            continue
        if dispersion(cx, cy, r) > 0.05:
            continue
        results.append((cx, cy, r, max(cov_n, cov_c)))

    results.sort(key=lambda t: -t[3])
    kept = []
    for cx, cy, r, _ in results:
        if any(math.hypot(cx - kx, cy - ky) < max(10.0, 0.05 * r) and
               abs(r - kr) < max(5.0, 0.05 * r) for kx, ky, kr in kept):
            continue
        kept.append((cx, cy, r))
    kept.sort(key=lambda t: -t[2])
    return [VectorPath(points=[], closed=True, kind="circle",
                       center=(cx / scale, cy / scale), radius=r / scale)
            for cx, cy, r in kept]

def detect_ellipses(gray: np.ndarray, mask: np.ndarray, scale: float,
                    maps=None) -> list[VectorPath]:
    """Détecte les ellipses et ovales (les « faux cercles ») comme entités exactes.

    Candidats : ajustement d'ellipse (cv2.fitEllipse) sur les contours du
    masque. Chaque candidat est amorcé sur les bords du trait (± demi-épaisseur
    estimée) puis affiné par descente de coordonnées bornée sur la couverture
    d'arêtes (centre, axes, inclinaison). Validation : couverture >= 0.7, plus
    grand secteur occlus <= 30 %, support du contour source sur l'ellipse
    (>= 80 %), isolement radial (proéminence au-delà de l'épaisseur locale du
    trait), dispersion radiale faible et épaisseur de trait uniforme le long
    du chemin (rejette pistes et blobs fusionnés). Les quasi-cercles (ratio
    d'axes > 0.92) sont laissés à detect_rings pour éviter les doublons ; les
    petites ellipses (grand axe < 5 % de l'image, glyphes de texte) sont
    rejetées par la taille minimale. Coords retournées : image originale.
    """
    dist, labels, ang, gmag, gblur = maps if maps is not None else build_ring_maps(gray)
    H, W = dist.shape
    max_dim = float(max(H, W))
    if np.count_nonzero(dist == 0) < 50:
        return []

    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    thr = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    crt, srt = np.cos(thr), np.sin(thr)

    def sample(cx, cy, a, b, tilt, cos_t, sin_t):
        tr = math.radians(tilt)
        ctr, str_ = math.cos(tr), math.sin(tr)
        ex = a * cos_t
        ey = b * sin_t
        px = cx + ex * ctr - ey * str_
        py = cy + ex * str_ + ey * ctr
        return (np.clip(np.round(px).astype(int), 0, W - 1),
                np.clip(np.round(py).astype(int), 0, H - 1))

    def cov_at(cx, cy, a, b, tilt, cos_t, sin_t):
        px, py = sample(cx, cy, a, b, tilt, cos_t, sin_t)
        return float((dist[py, px] <= 3.0).mean())

    def gap_of(hit):
        idx = np.nonzero(~hit)[0]
        n = len(hit)
        if len(idx) == 0:
            return 0.0
        if len(idx) == n:
            return 1.0
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        g = max(len(x) for x in runs) / n
        if not hit[0] and not hit[-1]:
            g = max(g, (len(runs[0]) + len(runs[-1])) / n)
        return g

    def width_profile(cx, cy, a, b, tilt):
        tsw = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        px0, py0 = sample(cx, cy, a, b, tilt, np.cos(tsw), np.sin(tsw))
        tr = math.radians(tilt)
        widths = []
        for i, (x0, y0) in enumerate(zip(px0, py0)):
            t = tsw[i]
            nx = b * math.cos(t) * math.cos(tr) - a * math.sin(t) * math.sin(tr)
            ny = b * math.cos(t) * math.sin(tr) + a * math.sin(t) * math.cos(tr)
            nn = math.hypot(nx, ny)
            if nn < 1e-9:
                continue
            nx, ny = nx / nn, ny / nn
            tot = 0.0
            for s in (1.0, -1.0):
                run = 0
                for step in range(1, 41):
                    x = int(round(x0 + s * step * nx))
                    y = int(round(y0 + s * step * ny))
                    if not (0 <= x < W and 0 <= y < H) or mask[y, x] == 0:
                        break
                    run = step
                tot += run
            widths.append(tot)
        return np.array(widths, dtype=float)

    def prom_of(cx, cy, a, b, tilt, center_cov, cos_t, sin_t):
        sw = float(np.median(width_profile(cx, cy, a, b, tilt)))
        rmean = 0.5 * (a + b)
        d0 = max(6.0, sw + 6.0)
        d1 = max(d0 + 4.0, 0.12 * rmean)
        offs = np.arange(d0, d1 + 0.1, max(2.0, (d1 - d0) / 7.0))
        sides = []
        for d in offs:
            fa = max(4.0, a - d)
            fb = max(4.0, b - d)
            ga = a + d
            gb = b + d
            sides.append(cov_at(cx, cy, fa, fb, tilt, cos_t, sin_t))
            sides.append(cov_at(cx, cy, ga, gb, tilt, cos_t, sin_t))
        return center_cov - max(sides)

    def dispersion(cx, cy, a, b, tilt):
        rmean = 0.5 * (a + b)
        n = int(min(1440, max(360, 2.0 * math.pi * rmean)))
        thd = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ctd, std_ = np.cos(thd), np.sin(thd)
        px0, py0 = sample(cx, cy, a, b, tilt, ctd, std_)
        vx, vy = px0 - cx, py0 - cy
        r0 = np.hypot(vx, vy)
        ux = vx / np.maximum(r0, 1e-9)
        uy = vy / np.maximum(r0, 1e-9)
        drs = np.arange(-4.0, 4.5, 1.0)
        stack = np.stack([
            dist[np.clip(np.round(cy + (r0 + dr) * uy).astype(int), 0, H - 1),
                 np.clip(np.round(cx + (r0 + dr) * ux).astype(int), 0, W - 1)]
            for dr in drs])
        best = np.argmin(stack, axis=0)
        dmin = stack[best, np.arange(n)]
        sel = dmin <= 3.0
        if int(sel.sum()) < 30:
            return 1.0
        err = drs[best[sel]]
        return float(err.std() / max(1.0, float(r0[sel].mean())))

    def refine(cx, cy, a, b, tilt):
        cov = cov_at(cx, cy, a, b, tilt, crt, srt)
        c0, a0, t0 = (cx, cy), a, tilt
        dc = max(3.0, 0.02 * a0)
        da = max(8.0, 0.08 * a0)
        moves = 0
        for step in (4.0, 2.0, 1.0):
            improved = True
            while improved and moves < 120:
                improved = False
                for idx in range(5):
                    for s in (step, -step):
                        params = [cx, cy, a, b, tilt]
                        params[idx] += s
                        if idx in (2, 3) and params[idx] < 4.0:
                            continue
                        if idx in (0, 1) and math.hypot(params[0] - c0[0], params[1] - c0[1]) > dc:
                            continue
                        if idx == 2 and abs(params[2] - a0) > da:
                            continue
                        if idx == 3 and abs(params[3] - a0) > da:
                            continue
                        if idx == 4 and min(abs(params[4] - t0), 180.0 - abs(params[4] - t0)) > 15.0:
                            continue
                        c2 = cov_at(params[0], params[1], params[2], params[3],
                                    params[4], crt, srt)
                        if c2 > cov + 1e-6:
                            cx, cy, a, b, tilt = params
                            cov = c2
                            improved = True
                            moves += 1
                            break
        return cx, cy, a, b, tilt, cov

    cands = []
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 40:
            continue
        try:
            (ecx, ecy), (d1, d2), eang = cv2.fitEllipse(pts[::2].astype(np.float32))
        except cv2.error:
            continue
        if d1 >= d2:
            a, b, tilt = d1 / 2.0, d2 / 2.0, float(eang)
        else:
            a, b, tilt = d2 / 2.0, d1 / 2.0, float(eang) + 90.0
        tilt %= 180.0
        if b < 0.012 * max_dim or a < 0.05 * max_dim or a > 0.8 * max_dim or b / a > 0.92:
            continue
        cands.append((float(ecx), float(ecy), a, b, tilt, pts))
    if not cands:
        return []

    results = []
    for cx, cy, a, b, tilt, cpts in cands:
        w0 = float(np.median(width_profile(cx, cy, a, b, tilt)))
        seeds = [(cx, cy, a, b, tilt),
                 (cx, cy, a + w0 / 2.0, b + w0 / 2.0, tilt),
                 (cx, cy, max(4.0, a - w0 / 2.0), max(4.0, b - w0 / 2.0), tilt)]
        refined = [refine(*seed) for seed in seeds]
        cx, cy, a, b, tilt, cov = max(refined, key=lambda r: r[5])
        tilt %= 180.0
        if not (0.012 * max_dim <= b and 0.05 * max_dim <= a <= 0.8 * max_dim):
            continue
        if b / a > 0.92:
            continue
        rmean = 0.5 * (a + b)
        dx = cpts[:, 0] - cx
        dy = cpts[:, 1] - cy
        trr = math.radians(tilt)
        dxr = dx * math.cos(trr) + dy * math.sin(trr)
        dyr = -dx * math.sin(trr) + dy * math.cos(trr)
        phi = np.arctan2(dyr, dxr)
        r_e = 1.0 / np.sqrt((np.cos(phi) / a) ** 2 + (np.sin(phi) / b) ** 2)
        err = np.abs(np.hypot(dxr, dyr) - r_e)
        if float((err <= max(4.0, 0.08 * rmean)).mean()) < 0.8:
            continue
        px, py = sample(cx, cy, a, b, tilt, ct, st)
        hits = dist[py, px] <= 3.0
        cov_n = float(hits.mean())
        if cov_n < 0.7 or gap_of(hits) > 0.30:
            continue
        if prom_of(cx, cy, a, b, tilt, cov_n, ct, st) < 0.10:
            continue
        if dispersion(cx, cy, a, b, tilt) > 0.05:
            continue
        wp = width_profile(cx, cy, a, b, tilt)
        on = wp > 0
        if float(on.mean()) < 0.55:
            continue
        wz = wp[on]
        if float((np.percentile(wz, 75) - np.percentile(wz, 25))
                / max(1.0, float(np.median(wz)))) > 0.6:
            continue
        results.append((cx, cy, a, b, tilt, cov_n))

    results.sort(key=lambda t: -t[5])
    kept = []
    for cx, cy, a, b, tilt, _ in results:
        rmean = 0.5 * (a + b)
        dup = False
        for kx, ky, ka, kb, kt in kept:
            krk = 0.5 * (ka + kb)
            if (math.hypot(cx - kx, cy - ky) < max(10.0, 0.05 * krk)
                    and abs(rmean - krk) < max(6.0, 0.07 * krk)):
                dup = True
                break
        if dup:
            continue
        kept.append((cx, cy, a, b, tilt))
    kept.sort(key=lambda t: -(t[2] * t[3]))
    return [VectorPath(points=[], closed=True, kind="ellipse",
                       center=(cx / scale, cy / scale), axis_a=a / scale, axis_b=b / scale,
                       tilt=tilt)
            for cx, cy, a, b, tilt in kept]


def _gap_to_curve(curve: VectorPath, pt) -> float:
    dx = pt[0] - curve.center[0]
    dy = pt[1] - curve.center[1]
    if curve.kind == "ellipse" and curve.axis_a > 0 and curve.axis_b > 0:
        tr = math.radians(curve.tilt)
        u = (dx * math.cos(tr) + dy * math.sin(tr)) / curve.axis_a
        v = (-dx * math.sin(tr) + dy * math.cos(tr)) / curve.axis_b
        return abs((math.hypot(u, v) - 1.0) * 0.5 * (curve.axis_a + curve.axis_b))
    return abs(math.hypot(dx, dy) - curve.radius)


def _clip_to_band(p_keep, p_drop, circles: list[VectorPath], band: float):
    a = np.asarray(p_keep, dtype=np.float64)
    b = np.asarray(p_drop, dtype=np.float64)
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        pt = a + (b - a) * mid
        if min(_gap_to_curve(c, pt) for c in circles) <= band:
            hi = mid
        else:
            lo = mid
    if lo <= 0.0:
        return None
    pt = a + (b - a) * lo
    return (float(pt[0]), float(pt[1]))


def strip_points_near_circles(paths: list[VectorPath], circles: list[VectorPath], band: float) -> list[VectorPath]:
    if not circles:
        return paths
    out = []
    for p in paths:
        if p.kind != "poly" or len(p.points) < 2:
            out.append(p)
            continue
        pts = np.array(p.points, dtype=np.float64)
        n = len(pts)
        keep = np.ones(n, dtype=bool)
        for c in circles:
            if c.kind == "ellipse" and c.axis_a > 0 and c.axis_b > 0:
                dx = pts[:, 0] - c.center[0]
                dy = pts[:, 1] - c.center[1]
                tr = math.radians(c.tilt)
                u = (dx * math.cos(tr) + dy * math.sin(tr)) / c.axis_a
                v = (-dx * math.sin(tr) + dy * math.cos(tr)) / c.axis_b
                q = np.hypot(u, v)
                rmean = 0.5 * (c.axis_a + c.axis_b)
                keep &= np.abs((q - 1.0) * rmean) > band
            else:
                d = np.hypot(pts[:, 0] - c.center[0], pts[:, 1] - c.center[1])
                keep &= np.abs(d - c.radius) > band
        if keep.all():
            out.append(p)
            continue
        idx = np.nonzero(keep)[0]
        if len(idx) == 0:
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
        if p.closed and len(segs) > 1 and 0 in segs[0] and (n - 1) in segs[-1]:
            segs[0] = segs[-1] + segs[0]
            segs.pop()
        for seg in segs:
            first, last = seg[0], seg[-1]
            head = tail = None
            prev_i = first - 1 if first > 0 else (n - 1 if p.closed else None)
            if prev_i is not None and not keep[prev_i]:
                head = _clip_to_band(pts[first], pts[prev_i], circles, band)
            next_i = last + 1 if last < n - 1 else (0 if p.closed else None)
            if next_i is not None and not keep[next_i]:
                tail = _clip_to_band(pts[last], pts[next_i], circles, band)
            seg_pts = [head] if head else []
            seg_pts += [(round(float(pts[i][0]), 2), round(float(pts[i][1]), 2)) for i in seg]
            if tail:
                seg_pts.append(tail)
            if len(seg_pts) >= 2:
                out.append(VectorPath(points=seg_pts, closed=False))
    return out


def remove_ring_shading(paths: list[VectorPath], circles: list[VectorPath], band: float, tol: float) -> list[VectorPath]:
    """Supprime le long des cercles parfaits les traits redondants : doublons (arc ou
    polyligne qui recopie un cercle deja detecte) et traits d'ombrage bosselles qui
    longent l'anneau sans former de forme propre. Un fragment circulaire bien fitte
    mais a un rayon/centre distinct est conserve."""
    if not circles:
        return paths
    out = []
    for p in paths:
        if p.kind in ("circle", "ellipse"):
            out.append(p)
            continue
        pts = np.array(curve_points(p), dtype=np.float64)
        drop = False
        for c in circles:
            r = c.radius
            W = max(band + 4.0, 0.06 * r)
            gaps = np.abs(np.hypot(pts[:, 0] - c.center[0], pts[:, 1] - c.center[1]) - r)
            if float((gaps <= W).mean()) < 0.7:
                continue
            fit = circle_fit(pts)
            same = (fit is not None
                    and math.hypot(fit[0] - c.center[0], fit[1] - c.center[1]) < 10
                    and abs(fit[2] - r) < max(band, 0.03 * r))
            wobbly = fit is None or fit[3] > max(2.0 * tol, 0.02 * r)
            if same or wobbly:
                drop = True
                break
        if not drop:
            out.append(p)
    return out


def curve_points(p: VectorPath, step_deg: float = 5.0) -> list:
    if p.kind == "circle":
        n = max(24, int(360 / step_deg))
        return [(p.center[0] + p.radius * math.cos(math.radians(360 * i / n)),
                 p.center[1] + p.radius * math.sin(math.radians(360 * i / n))) for i in range(n)]
    if p.kind == "ellipse":
        n = max(24, int(360 / step_deg))
        tr = math.radians(p.tilt)
        ctr, str_ = math.cos(tr), math.sin(tr)
        pts = []
        for i in range(n):
            t = math.radians(360.0 * i / n)
            ex, ey = p.axis_a * math.cos(t), p.axis_b * math.sin(t)
            pts.append((p.center[0] + ex * ctr - ey * str_,
                        p.center[1] + ex * str_ + ey * ctr))
        return pts
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

        if self.fit_curves:
            paths = detect_curves(paths, tol / scale, max(w0, h0))
        closed_count = sum(1 for p in paths if p.closed)
        if self.close_lines:
            gap = self.gap if self.gap is not None else max(1.5, 0.004 * max(w0, h0))
            paths = close_paths(paths, gap, self.closed_only)
            closed_count = sum(1 for p in paths if p.closed)
        if self.fit_curves:
            maps = build_ring_maps(gray)
            verified = []
            for p in paths:
                if p.kind == "circle":
                    cov_n, cov_c, gap_n, step = ring_scores(maps,
                                                            p.center[0] * scale, p.center[1] * scale,
                                                            p.radius * scale)
                    if not ((cov_n >= 0.7 or cov_c >= 0.55) and gap_n <= 0.3 and step >= 30.0):
                        p = VectorPath(points=curve_points(p), closed=True)
                verified.append(p)
            paths = verified
            ring_circles = detect_rings(gray, mask, scale, maps=maps)
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
            ellipses = detect_ellipses(gray, mask, scale, maps=maps)
            new_ellipses = []
            for e in ellipses:
                clash = False
                for o in [p for p in paths if p.kind in ("circle", "ellipse")] + new_circles:
                    if o.kind == "circle":
                        same = (abs(e.axis_a - o.radius) < max(stroke_up / scale, 4.0)
                                and abs(e.axis_b - o.radius) < max(stroke_up / scale, 4.0))
                    else:
                        same = (abs(e.axis_a - o.axis_a) < max(stroke_up / scale, 4.0)
                                and abs(e.axis_b - o.axis_b) < max(stroke_up / scale, 4.0))
                    if (math.hypot(e.center[0] - o.center[0], e.center[1] - o.center[1]) < 10
                            and same):
                        clash = True
                        break
                if clash:
                    continue
                ea = sorted((e.axis_a, e.axis_b))
                if any(math.hypot(e.center[0] - n.center[0], e.center[1] - n.center[1]) < 8
                       and abs(ea[0] - na) < max(3.0, 0.02 * na)
                       and abs(ea[1] - nb) < max(3.0, 0.02 * nb)
                       for n in new_ellipses
                       for na, nb in [sorted((n.axis_a, n.axis_b))]):
                    continue
                new_ellipses.append(e)
            if new_circles or new_ellipses:
                all_curves = ([p for p in paths if p.kind in ("circle", "ellipse")]
                              + new_circles + new_ellipses)
                band = max(2.5, 0.75 * stroke_up / scale)
                paths = strip_points_near_circles(paths, all_curves, band) + new_circles + new_ellipses
                paths = remove_ring_shading(paths,
                                            [c for c in all_curves if c.kind == "circle"],
                                            band, tol / scale)
        n_circles = sum(1 for p in paths if p.kind == "circle")
        n_arcs = sum(1 for p in paths if p.kind == "arc")
        n_ellipses = sum(1 for p in paths if p.kind == "ellipse")

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
            "ellipses": n_ellipses,
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
