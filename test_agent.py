#!/usr/bin/env python3
"""Tests unitaires de l'agent analyste d'image.

Lancement :
    python -m unittest test_agent -v
"""
from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from agent import (
    QUALITY_TOLERANCE,
    COLOR_NAMES,
    _drop_gradient_edges,
    ImageVectorAgent,
    VectorPath,
    binarize,
    build_ring_maps,
    circle_fit,
    close_paths,
    detect_curves,
    detect_ellipses,
    detect_rings,
    dominant_color,
    load_image,
    preprocess,
    quantize_colors,
    remove_ring_shading,
    remove_specks,
    ring_scores,
    strip_points_near_circles,
    write_dxf,
    write_svg,
)

BLACK = (20, 20, 20)


def make_canvas(size: int, noise: float = 4.0, seed: int = 3) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, noise, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def prep(gray: np.ndarray, invert: bool = False):
    mask, _ = binarize(gray, "auto", invert)
    mask = remove_specks(mask, 2e-5)
    maps = build_ring_maps(gray)
    return mask, maps


def circles_of(paths) -> list:
    return [(p.center[0], p.center[1], p.radius) for p in paths if p.kind == "circle"]


def detect_on_canvas(img: np.ndarray) -> list:
    """Chemin réel : écriture JPEG, relecture, prétraitement, détection.
    Coords retournées dans le repère du canvas fourni."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "canvas.jpg"
        cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        loaded = load_image(src)
    gray0 = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
    gray, scale = preprocess(gray0)
    mask, _ = binarize(gray, "auto", False)
    mask = remove_specks(mask, 2e-5)
    maps = build_ring_maps(gray)
    return detect_rings(gray, mask, scale, maps=maps) + detect_ellipses(gray, mask, scale, maps=maps)


def has_circle(circles, cx, cy, r, tol_c, tol_r) -> bool:
    return any(abs(x - cx) <= tol_c and abs(y - cy) <= tol_c and abs(rr - r) <= tol_r
               for x, y, rr in circles)


class TestOutilsGeometrie(unittest.TestCase):
    def test_circle_fit_cercle_parfait(self):
        th = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        pts = np.column_stack([200.0 + 100.0 * np.cos(th), 200.0 + 100.0 * np.sin(th)])
        cx, cy, r, err = circle_fit(pts)
        self.assertAlmostEqual(cx, 200.0, places=3)
        self.assertAlmostEqual(cy, 200.0, places=3)
        self.assertAlmostEqual(r, 100.0, places=3)
        self.assertLess(err, 1e-6)

    def test_preprocess_agrandit_les_petites_images(self):
        gray = np.full((200, 400), 255, dtype=np.uint8)
        up, scale = preprocess(gray)
        self.assertEqual(max(up.shape[:2]), 1600)
        self.assertAlmostEqual(scale, 4.0)

    def test_qualite_tolerance_croissante(self):
        self.assertEqual(set(QUALITY_TOLERANCE), {"max", "high", "balanced"})
        self.assertLess(QUALITY_TOLERANCE["max"], QUALITY_TOLERANCE["high"])
        self.assertLess(QUALITY_TOLERANCE["high"], QUALITY_TOLERANCE["balanced"])


class TestRingScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = make_canvas(800)
        cv2.circle(img, (400, 400), 200, BLACK, -1)
        cls.gray = to_gray(img)
        cls.maps = build_ring_maps(cls.gray)

    def test_vrai_cercle_bien_note(self):
        cov_n, cov_c, gap, step = ring_scores(self.maps, 400.0, 400.0, 200.0)
        self.assertGreater(cov_n, 0.9)
        self.assertLess(gap, 0.05)
        self.assertGreater(step, 30.0)

    def test_mauvais_rayon_mal_note(self):
        cov_n, _, _, _ = ring_scores(self.maps, 400.0, 400.0, 230.0)
        self.assertLess(cov_n, 0.5)


class TestDetectRings(unittest.TestCase):
    def detect(self, img):
        return circles_of(detect_on_canvas(img))

    def test_cercle_complet(self):
        img = make_canvas(800)
        cv2.circle(img, (400, 400), 200, BLACK, 6)
        cs = self.detect(img)
        self.assertTrue(has_circle(cs, 400, 400, 200, 5, 5))

    def test_cercle_occulte(self):
        img = make_canvas(800)
        cv2.ellipse(img, (400, 400), (200, 200), 0, 90, 390, BLACK, 6)
        cs = self.detect(img)
        self.assertTrue(has_circle(cs, 400, 400, 200, 5, 5))

    def test_double_piste_concentrique(self):
        img = make_canvas(800)
        cv2.circle(img, (400, 400), 140, BLACK, 3)
        cv2.circle(img, (400, 400), 159, BLACK, 3)
        cs = self.detect(img)
        self.assertTrue(has_circle(cs, 400, 400, 140, 5, 5))
        self.assertTrue(has_circle(cs, 400, 400, 159, 5, 5))

    def test_grand_et_petit_cercles(self):
        img = make_canvas(800)
        cv2.circle(img, (300, 400), 220, BLACK, 6)
        cv2.circle(img, (600, 250), 20, BLACK, 4)
        cs = self.detect(img)
        self.assertTrue(has_circle(cs, 300, 400, 220, 5, 5))
        self.assertTrue(has_circle(cs, 600, 250, 20, 5, 5))

    def test_ellipse_rejetee(self):
        img = make_canvas(800)
        cv2.ellipse(img, (300, 400), (150, 75), 0, 0, 360, BLACK, 5)
        cs = self.detect(img)
        self.assertFalse(any(math.hypot(x - 300.0, y - 400.0) < 30.0 for x, y, _ in cs))

    def test_glyphe_texte_rejete(self):
        img = make_canvas(800)
        cv2.ellipse(img, (200, 200), (24, 16), 0, 0, 360, BLACK, 2)
        cv2.ellipse(img, (500, 600), (20, 14), 30, 0, 360, BLACK, 2)
        cs = self.detect(img)
        self.assertFalse(any(math.hypot(x - 200.0, y - 200.0) < 20.0 for x, y, _ in cs))
        self.assertFalse(any(math.hypot(x - 500.0, y - 600.0) < 20.0 for x, y, _ in cs))

    def test_performance_sous_limite(self):
        img = make_canvas(1600)
        cv2.circle(img, (800, 800), 500, BLACK, 6)
        cv2.circle(img, (1200, 400), 40, BLACK, 5)
        t0 = time.time()
        cs = self.detect(img)
        duree = time.time() - t0
        self.assertTrue(has_circle(cs, 800, 800, 500, 6, 10))
        self.assertTrue(has_circle(cs, 1200, 400, 40, 5, 5))
        self.assertLess(duree, 20.0)


class TestDetectEllipses(unittest.TestCase):
    def ellipses_of(self, paths) -> list:
        return [(p.center[0], p.center[1], p.axis_a, p.axis_b, p.tilt)
                for p in paths if p.kind == "ellipse"]

    def has_ellipse(self, els, cx, cy, a, b, tilt, tol) -> bool:
        return any(abs(x - cx) <= tol and abs(y - cy) <= tol and abs(aa - a) <= tol
                   and abs(bb - b) <= tol
                   and min(abs(t - tilt), 180.0 - abs(t - tilt)) <= 6.0
                   for x, y, aa, bb, t in els)

    def test_ellipse_detectee(self):
        img = make_canvas(800)
        cv2.ellipse(img, (400, 400), (150, 75), 0, 0, 360, BLACK, 5)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertTrue(self.has_ellipse(els, 400, 400, 150, 75, 0, 6))

    def test_ellipse_inclinee(self):
        img = make_canvas(800)
        cv2.ellipse(img, (400, 400), (150, 75), 30, 0, 360, BLACK, 5)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertTrue(self.has_ellipse(els, 400, 400, 150, 75, 30, 6))

    def test_ellipse_occultee(self):
        img = make_canvas(800)
        cv2.ellipse(img, (400, 400), (150, 75), 0, 0, 300, BLACK, 5)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertTrue(self.has_ellipse(els, 400, 400, 150, 75, 0, 6))

    def test_quasi_cercle_pas_ellipse(self):
        img = make_canvas(800)
        cv2.ellipse(img, (400, 400), (150, 140), 0, 0, 360, BLACK, 5)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertEqual(els, [])

    def test_cercle_pas_ellipse(self):
        img = make_canvas(800)
        cv2.circle(img, (400, 400), 150, BLACK, 5)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertEqual(els, [])

    def test_ligne_droite_pas_ellipse(self):
        img = make_canvas(800)
        cv2.line(img, (100, 100), (700, 700), BLACK, 4)
        els = self.ellipses_of(detect_on_canvas(img))
        self.assertEqual(els, [])


class TestDetectCurves(unittest.TestCase):
    def test_polygone_circulaire_devient_cercle(self):
        th = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        pts = [(200.0 + 100.0 * math.cos(t), 200.0 + 100.0 * math.sin(t)) for t in th]
        res = detect_curves([VectorPath(points=pts, closed=True)], 0.5, 800.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].kind, "circle")
        self.assertAlmostEqual(res[0].center[0], 200.0, delta=0.5)
        self.assertAlmostEqual(res[0].radius, 100.0, delta=0.5)

    def test_arc_detecte(self):
        th = np.linspace(math.radians(20.0), math.radians(140.0), 30)
        pts = [(200.0 + 100.0 * math.cos(t), 200.0 + 100.0 * math.sin(t)) for t in th]
        res = detect_curves([VectorPath(points=pts, closed=False)], 0.5, 800.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].kind, "arc")
        self.assertAlmostEqual(abs(res[0].sweep), 120.0, delta=5.0)
        self.assertAlmostEqual(res[0].radius, 100.0, delta=0.5)

    def test_ligne_brisee_restee_poly(self):
        pts = [(100.0, 100.0), (300.0, 140.0), (180.0, 260.0), (350.0, 300.0)]
        res = detect_curves([VectorPath(points=pts, closed=False)], 0.5, 800.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].kind, "poly")


class TestClosePaths(unittest.TestCase):
    def test_carre_en_4_segments_ferme(self):
        seg = [
            [(100, 100), (294, 100)],
            [(300, 106), (300, 294)],
            [(294, 300), (106, 300)],
            [(100, 294), (100, 106)],
        ]
        paths = [VectorPath(points=s, closed=False) for s in seg]
        res = close_paths(paths, 12.0, False)
        self.assertGreaterEqual(sum(1 for p in res if p.closed), 1)

    def test_lignes_eloignees_restent_ouvertes(self):
        paths = [
            VectorPath(points=[(100, 100), (300, 100)], closed=False),
            VectorPath(points=[(100, 400), (300, 400)], closed=False),
        ]
        res = close_paths(paths, 12.0, False)
        self.assertFalse(any(p.closed for p in res))
        self.assertEqual(len(res), 2)

    def test_closed_only_supprime_les_ouverts(self):
        paths = [
            VectorPath(points=[(100, 100), (294, 100)], closed=False),
            VectorPath(points=[(300, 106), (300, 294)], closed=False),
            VectorPath(points=[(294, 300), (106, 300)], closed=False),
            VectorPath(points=[(100, 294), (100, 106)], closed=False),
        ]
        res = close_paths(paths, 12.0, True)
        self.assertGreaterEqual(len(res), 1)
        self.assertTrue(all(p.closed for p in res))


class TestStripPointsNearCircles(unittest.TestCase):
    def test_points_du_cercle_retires_autre_conserve(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        th = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
        sur_cercle = [(200.0 + 100.0 * math.cos(t), 200.0 + 100.0 * math.sin(t)) for t in th]
        loin = [(400.0, 500.0), (700.0, 500.0)]
        res = strip_points_near_circles(
            [VectorPath(points=sur_cercle, closed=False), VectorPath(points=loin, closed=False)],
            [cercle], 5.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].points, loin)

    def test_trait_arrivant_sur_cercle_conserve_et_raccourci(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        trait = VectorPath(points=[(200.0, 296.0), (200.0, 330.0)], closed=False)
        res = strip_points_near_circles([trait], [cercle], 5.0)
        self.assertEqual(len(res), 1)
        pts = res[0].points
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[1][0], 200.0, places=1)
        self.assertAlmostEqual(pts[1][1], 330.0, places=1)
        d0 = math.hypot(pts[0][0] - 200.0, pts[0][1] - 200.0)
        self.assertAlmostEqual(d0, 105.0, delta=0.6)

    def test_trait_entierement_dans_le_bandeau_supprime(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        trait = VectorPath(points=[(200.0, 298.0), (200.0, 302.0)], closed=False)
        res = strip_points_near_circles([trait], [cercle], 5.0)
        self.assertEqual(len(res), 0)

    def test_trait_traversant_le_cercle_coupe_aux_deux_bords(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        trait = VectorPath(points=[(200.0, 150.0), (200.0, 298.0), (200.0, 320.0)], closed=False)
        res = strip_points_near_circles([trait], [cercle], 5.0)
        self.assertEqual(len(res), 2)
        for seg in res:
            self.assertEqual(len(seg.points), 2)
        ys = sorted(pt[1] for seg in res for pt in seg.points)
        self.assertAlmostEqual(ys[0], 150.0, places=1)
        self.assertAlmostEqual(ys[1], 295.0, delta=0.6)
        self.assertAlmostEqual(ys[2], 305.0, delta=0.6)
        self.assertAlmostEqual(ys[3], 320.0, places=1)

    def test_trait_arrivant_sur_ellipse_conserve(self):
        ell = VectorPath(points=[], closed=True, kind="ellipse",
                         center=(200.0, 200.0), axis_a=120.0, axis_b=60.0, tilt=0.0)
        trait = VectorPath(points=[(320.0, 200.0), (360.0, 200.0)], closed=False)
        res = strip_points_near_circles([trait], [ell], 5.0)
        self.assertEqual(len(res), 1)
        pts = res[0].points
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[1][0], 360.0, places=1)
        d0 = math.hypot(pts[0][0] - 200.0, pts[0][1] - 200.0)
        self.assertAlmostEqual(d0, 126.67, delta=0.6)


class TestRemoveRingShading(unittest.TestCase):
    def test_doublon_arc_supprime(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        th = np.linspace(math.radians(10), math.radians(120), 30)
        arc = VectorPath(
            points=[(200.0 + 100.0 * math.cos(t), 200.0 + 100.0 * math.sin(t)) for t in th],
            closed=False, kind="arc", center=(200.0, 200.0), radius=100.0,
            start_angle=10.0, sweep=110.0)
        res = remove_ring_shading([arc], [cercle], 4.0, 0.8)
        self.assertEqual(len(res), 0)

    def test_ombrage_bossule_supprime(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        th = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
        wob = [(200.0 + (105.0 + 3.0 * math.sin(5.0 * t)) * math.cos(t),
                200.0 + (105.0 + 3.0 * math.sin(5.0 * t)) * math.sin(t)) for t in th]
        res = remove_ring_shading([VectorPath(points=wob, closed=False)], [cercle], 4.0, 0.8)
        self.assertEqual(len(res), 0)

    def test_arc_distinct_conserve(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        th = np.linspace(math.radians(10), math.radians(120), 30)
        arc = VectorPath(
            points=[(200.0 + 60.0 * math.cos(t), 200.0 + 60.0 * math.sin(t)) for t in th],
            closed=False, kind="arc", center=(200.0, 200.0), radius=60.0,
            start_angle=10.0, sweep=110.0)
        res = remove_ring_shading([arc], [cercle], 4.0, 0.8)
        self.assertEqual(len(res), 1)

    def test_trait_traversant_conserve(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(200.0, 200.0), radius=100.0)
        ligne = VectorPath(points=[(0.0, 200.0), (400.0, 200.0)], closed=False)
        res = remove_ring_shading([ligne], [cercle], 4.0, 0.8)
        self.assertEqual(len(res), 1)


class TestFiltreDegrades(unittest.TestCase):
    def test_marche_faible_lointaine_supprimee(self):
        smooth = np.full((60, 60), 100, dtype=np.uint8)
        smooth[:, :30] = 30
        smooth[:, 30:] = 200
        smooth[50:60, 35:55] = 180
        edges = np.zeros((60, 60), np.uint8)
        edges[10:45, 30] = 255
        edges[50, 35:55] = 255
        res = _drop_gradient_edges(edges, smooth)
        self.assertGreaterEqual((res[10:45, 30] > 0).sum(), 30)
        self.assertEqual((res[50, 35:55] > 0).sum(), 0)

    def test_marche_faible_raccordee_conservee(self):
        smooth = np.full((60, 60), 100, dtype=np.uint8)
        smooth[:, :30] = 30
        smooth[:30, 30:] = 200
        smooth[30:, 30:] = 60
        edges = np.zeros((60, 60), np.uint8)
        edges[10:40, 30] = 255
        res = _drop_gradient_edges(edges, smooth)
        self.assertGreaterEqual((res[10:31, 30] > 0).sum(), 20)
        self.assertEqual((res[33:40, 30] > 0).sum(), 0)


class TestExports(unittest.TestCase):
    def test_write_svg_cercle(self):
        p = VectorPath(points=[], closed=True, kind="circle", center=(100.0, 200.0), radius=50.0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.svg"
            write_svg([p], 400, 400, 1.0, out)
            txt = out.read_text(encoding="utf-8")
        self.assertIn('<circle cx="100.00" cy="200.00" r="50.00"/>', txt)
        self.assertIn('fill="none"', txt)

    def test_write_svg_ellipse(self):
        p = VectorPath(points=[], closed=True, kind="ellipse", center=(100.0, 200.0),
                       axis_a=80.0, axis_b=40.0, tilt=30.0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.svg"
            write_svg([p], 400, 400, 1.0, out)
            txt = out.read_text(encoding="utf-8")
        self.assertIn('<ellipse cx="100.00" cy="200.00" rx="80.00" ry="40.00"', txt)
        self.assertIn('rotate(30.00 100.00 200.00)', txt)

    def test_write_dxf_entite_ellipse(self):
        p = VectorPath(points=[], closed=True, kind="ellipse", center=(100.0, 200.0),
                       axis_a=80.0, axis_b=40.0, tilt=0.0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.dxf"
            write_dxf([p], 400, "px", 1.0, out)
            doc = ezdxf.readfile(out)
        e = next(e for e in doc.modelspace() if e.dxftype() == "ELLIPSE")
        self.assertAlmostEqual(e.dxf.center[0], 100.0, places=4)
        self.assertAlmostEqual(e.dxf.center[1], 200.0, places=4)
        self.assertAlmostEqual(e.dxf.major_axis[0], 80.0, places=4)
        self.assertAlmostEqual(e.dxf.major_axis[1], 0.0, places=4)
        self.assertAlmostEqual(e.dxf.ratio, 0.5, places=4)

    def test_write_dxf_entites_cao(self):
        cercle = VectorPath(points=[], closed=True, kind="circle", center=(100.0, 200.0), radius=50.0)
        arc = VectorPath(points=[], closed=False, kind="arc", center=(300.0, 200.0), radius=40.0,
                         start_angle=0.0, sweep=90.0)
        carre = VectorPath(points=[(10, 10), (60, 10), (60, 60), (10, 60)], closed=True)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.dxf"
            write_dxf([cercle, arc, carre], 400, "px", 1.0, out)
            doc = ezdxf.readfile(out)
        types = [e.dxftype() for e in doc.modelspace()]
        self.assertEqual(types.count("CIRCLE"), 1)
        self.assertEqual(types.count("ARC"), 1)
        self.assertEqual(types.count("LWPOLYLINE"), 1)
        c = next(e for e in doc.modelspace() if e.dxftype() == "CIRCLE")
        self.assertAlmostEqual(c.dxf.center[0], 100.0, places=4)
        self.assertAlmostEqual(c.dxf.center[1], 200.0, places=4)
        self.assertAlmostEqual(c.dxf.radius, 50.0, places=4)


class TestPipelineComplet(unittest.TestCase):
    def test_run_bout_en_bout(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "synthetic.jpg"
            outdir = Path(d) / "out"
            img = make_canvas(800)
            cv2.rectangle(img, (80, 80), (720, 720), BLACK, 3)
            cv2.circle(img, (400, 400), 180, BLACK, 3)
            cv2.ellipse(img, (400, 400), (120, 60), 25, 0, 360, BLACK, 3)
            cv2.line(img, (80, 80), (220, 220), BLACK, 2)
            cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            agent = ImageVectorAgent(quality="high")
            report = agent.run(src, outdir)
            self.assertGreaterEqual(report["circles"], 1)
            self.assertGreaterEqual(report["ellipses"], 1)
            self.assertGreater(report["paths"], 0)
            for key in ("svg", "dxf", "preview", "report"):
                self.assertTrue(Path(report["outputs"][key]).exists())
            svg = Path(report["outputs"]["svg"]).read_text(encoding="utf-8")
            self.assertIn("<circle", svg)
            doc = ezdxf.readfile(report["outputs"]["dxf"])
            self.assertGreaterEqual(
                sum(1 for e in doc.modelspace() if e.dxftype() == "CIRCLE"), 1)
    def test_arcs_conserves_avec_lignes_fermees(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "synthetic.jpg"
            outdir = Path(d) / "out"
            img = make_canvas(800)
            cv2.circle(img, (400, 250), 120, BLACK, 3)
            cv2.ellipse(img, (400, 600), (100, 100), 0, 200.0, 320.0, BLACK, 3)
            cv2.line(img, (80, 700), (220, 700), BLACK, 2)
            cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            agent = ImageVectorAgent(quality="high", close_lines=True)
            report = agent.run(src, outdir)
            self.assertGreaterEqual(report["arcs"], 1)
            doc = ezdxf.readfile(report["outputs"]["dxf"])
            self.assertGreaterEqual(
                sum(1 for e in doc.modelspace() if e.dxftype() == "ARC"), 1)


class TestCouleurs(unittest.TestCase):
    def test_quantize_colors_six_zones(self):
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        zones = {
            "rouge": (60, 60, 220),
            "jaune": (50, 220, 240),
            "vert": (60, 200, 60),
            "bleu": (220, 80, 40),
            "blanc": (245, 245, 245),
            "noir": (20, 20, 20),
        }
        for k, name in enumerate(COLOR_NAMES):
            col = k % 3
            row = k // 3
            img[row * 150:(row + 1) * 150, col * 200:(col + 1) * 200] = zones[name]
        labels = quantize_colors(img)
        for k, name in enumerate(COLOR_NAMES):
            col = k % 3
            row = k // 3
            zone = labels[row * 150 + 40:(row + 1) * 150 - 40, col * 200 + 40:(col + 1) * 200 - 40]
            self.assertEqual(int(np.bincount(zone.ravel(), minlength=6).argmax()), k,
                             f"zone {name} mal classee")

    def test_dominant_color_vote(self):
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[:, 50:] = 2
        self.assertEqual(dominant_color(labels, [(10, 10), (20, 30)]), "bleu")
        self.assertEqual(dominant_color(labels, [(60, 10), (70, 30), (80, 20)]), "rouge")
        self.assertEqual(dominant_color(labels, []), "")

    def test_run_couleurs_bout_en_bout(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "couleur.jpg"
            outdir = Path(d) / "out"
            img = make_canvas(800)
            cv2.rectangle(img, (100, 100), (300, 300), (60, 60, 220), -1)
            cv2.circle(img, (550, 400), 120, (50, 220, 240), -1)
            cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            agent = ImageVectorAgent(quality="high", colors="rouge,jaune")
            report = agent.run(src, outdir)
            self.assertGreaterEqual(report["circles"], 1)
            self.assertGreater(report["paths"], 0)
            self.assertTrue(any(v > 0 for v in report["colors"].values()))
            svg = Path(report["outputs"]["svg"]).read_text(encoding="utf-8")
            self.assertIn("#d62728", svg)
            self.assertIn("#e8b923", svg)
            self.assertIn('#fafafa', svg)
            doc = ezdxf.readfile(report["outputs"]["dxf"])
            layers = {l.dxf.name for l in doc.layers}
            self.assertIn("COL_ROUGE", layers)
            self.assertIn("COL_JAUNE", layers)

    def test_selection_une_seule_couleur(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "couleur.jpg"
            outdir = Path(d) / "out"
            img = make_canvas(800)
            cv2.rectangle(img, (100, 100), (300, 300), (60, 60, 220), -1)
            cv2.circle(img, (550, 400), 120, (50, 220, 240), -1)
            cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            agent = ImageVectorAgent(quality="high", colors="rouge")
            report = agent.run(src, outdir)
            svg = Path(report["outputs"]["svg"]).read_text(encoding="utf-8")
            self.assertIn("#d62728", svg)
            self.assertNotIn("#e8b923", svg)

    def test_couleur_inconnue_erreur(self):
        with self.assertRaises(ValueError):
            ImageVectorAgent(colors="pourpre")
        with self.assertRaises(ValueError):
            ImageVectorAgent(colors="rouge,pourpre")
        self.assertIsNone(ImageVectorAgent._parse_colors("none"))
        self.assertIsNone(ImageVectorAgent._parse_colors(""))
        self.assertEqual(ImageVectorAgent._parse_colors("toutes"), list(COLOR_NAMES))
        self.assertEqual(ImageVectorAgent._parse_colors("rouge; jaune"), ["rouge", "jaune"])


if __name__ == "__main__":
    unittest.main()
