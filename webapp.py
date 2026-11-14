"""Interface web de l'agent analyste d'image : upload -> SVG + DXF.

Usage:
    python webapp.py [--host 127.0.0.1] [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

from agent import ImageVectorAgent

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
JOBS_DIR = Path(__file__).resolve().parent / "jobs"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

HEAD = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent analyste d'image</title>
<style>
  body { font-family: system-ui, sans-serif; background: #f4f5f7; margin: 0; }
  .wrap { max-width: 720px; margin: 2.5rem auto; padding: 0 1rem; }
  .card { background: #fff; border-radius: 12px; padding: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  h1 { font-size: 1.4rem; margin-top: 0; }
  label { display: block; margin: 1rem 0 .3rem; font-weight: 600; font-size: .9rem; }
  select, input[type=text] { width: 100%; padding: .5rem; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; }
  input[type=file] { margin-top: .3rem; }
  .row { display: flex; gap: 1rem; }
  .row > div { flex: 1; }
  button { margin-top: 1.5rem; width: 100%; padding: .8rem; font-size: 1rem; font-weight: 600;
           color: #fff; background: #1a73e8; border: 0; border-radius: 8px; cursor: pointer; }
  button:hover { background: #1557b0; }
  .error { background: #fdecea; color: #b3261e; padding: .8rem 1rem; border-radius: 8px; margin-bottom: 1rem; }
  .preview { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; margin: 1rem 0; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; margin: 1rem 0; }
  td, th { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; }
  th { color: #555; font-weight: 600; width: 45%; }
  .btn { display: inline-block; margin: .3rem .5rem .3rem 0; padding: .6rem 1.2rem; font-weight: 600;
         text-decoration: none; border-radius: 8px; background: #1a73e8; color: #fff; }
  .btn.secondary { background: #e8eaed; color: #1a1a1a; }
  a.back { display: inline-block; margin-top: 1.5rem; color: #1a73e8; }
  .curseurs { display: grid; grid-template-columns: 1fr 1fr; gap: .4rem 1.5rem; margin: 1rem 0; }
  .curseurs label { margin: .4rem 0 .1rem; font-size: .85rem; }
  .curseurs label span { color: #1a73e8; font-weight: 700; }
  input[type=range] { width: 100%; }
  .apercu { border: 1px solid #ddd; border-radius: 8px; background: #fafafa;
            max-height: 70vh; overflow: auto; margin: 1rem 0; text-align: center; }
  .apercu svg { display: block; margin: 0 auto; }
  #status { font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
"""

FOOT = """</div>
</body>
</html>"""

FORM_BODY = """<div class="card">
  <h1>Agent analyste d'image</h1>
  <p>Chargez une image (JPEG, PNG, BMP, WebP, TIFF, 32 Mo max). L'agent la convertit
     en tracé vectoriel : un fichier <strong>SVG</strong> fidèle et un fichier <strong>DXF</strong> pour la CAO.</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="{{ url_for('convert') }}" enctype="multipart/form-data">
    <label for="image">Image à convertir</label>
    <input type="file" id="image" name="image" accept=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff" required>
    <div class="row">
      <div>
        <label for="mode">Mode d'extraction</label>
        <select id="mode" name="mode">
          <option value="auto" selected>Automatique</option>
          <option value="centerline">Tracé central (plans, dessins au trait)</option>
          <option value="outline">Contours (logos, formes pleines)</option>
          <option value="edges">Détails internes (logos détaillés, photos)</option>
        </select>
      </div>
      <div>
        <label for="quality">Qualité du tracé</label>
        <select id="quality" name="quality">
          <option value="max">Maximum (plus de points)</option>
          <option value="high" selected>Haute (recommandé)</option>
          <option value="balanced">Équilibrée (fichier léger)</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div>
        <label for="unit">Unités du DXF</label>
        <select id="unit" name="unit">
          <option value="px" selected>Pixels</option>
          <option value="mm">Millimètres</option>
          <option value="cm">Centimètres</option>
          <option value="in">Pouces</option>
        </select>
      </div>
      <div>
        <label for="scale">Échelle DXF (1 px = … unités)</label>
        <input type="text" id="scale" name="scale" value="1.0" inputmode="decimal">
      </div>
    </div>
    <label style="font-weight:400">
      <input type="checkbox" name="invert" value="1" style="width:auto">
      Tracé clair sur fond sombre
    </label>
    <label style="font-weight:400">
      <input type="checkbox" name="close" value="1" style="width:auto" checked>
      Tracés fermés (gravure / découpe laser)
    </label>
    <label style="font-weight:400">
      <input type="checkbox" name="curves" value="1" style="width:auto" checked>
      Cercles et arcs parfaits (CAO)
    </label>
    <div>
      <label for="colors">Couleurs à tracer</label>
      <input type="text" id="colors" name="colors" placeholder="ex : rouge, jaune — ou 'toutes' (vide = noir et blanc)">
    </div>
    <div class="row">
      <div>
        <label for="stroke_width">Épaisseur du trait SVG (px)</label>
        <input type="text" id="stroke_width" name="stroke_width" value="1" inputmode="decimal">
      </div>
      <div>
        <label>&nbsp;</label>
        <span style="font-size:.8rem;color:#666">Laisser « auto » pour l'épaisseur estimée</span>
      </div>
    </div>
    <button type="submit">Convertir</button>
  </form>
</div>"""

RESULT_BODY = """<div class="card">
  <h1>Résultat de la conversion</h1>
  <p id="stats"><strong>{{ report.source|basename }}</strong> — {{ report.analysis.width }}×{{ report.analysis.height }} px,
     mode {{ report.analysis.mode_used }}, seuillage {{ report.analysis.threshold_method }},
     {{ report.paths }} tracés (dont {{ report.closed_paths }} fermés, {{ report.circles }} cercles,
     {{ report.get('ellipses', 0) }} ellipses et {{ report.arcs }} arcs parfaits), {{ report.points }} points.</p>
  <div class="curseurs">
    <div>
      <label for="zoom">Zoom <span id="zoom_val">100</span> %</label>
      <input type="range" id="zoom" min="30" max="400" step="10" value="100">
    </div>
    <div>
      <label for="sw">Épaisseur du trait <span id="sw_val">{{ init_sw }}</span> px <em>(immédiat)</em></label>
      <input type="range" id="sw" min="0.25" max="6" step="0.05" value="{{ init_sw }}">
    </div>
    <div>
      <label for="tol">Tolérance de simplification <span id="tol_val">{{ init_tol }}</span> px</label>
      <input type="range" id="tol" min="0.45" max="3" step="0.05" value="{{ init_tol }}">
    </div>
    <div>
      <label for="gap">Raccord des extrémités <span id="gap_val">{{ init_gap }}</span> px (0 = pas de fermeture)</label>
      <input type="range" id="gap" min="0" max="10" step="0.5" value="{{ init_gap }}">
    </div>
  </div>
  <div>
    <label for="live_colors">Couleurs à tracer</label>
    <input type="text" id="live_colors" value="{{ init_colors }}" placeholder="ex : rouge, jaune — ou 'toutes' (vide = noir et blanc)">
  </div>
  <p style="font-size:.85rem;color:#666">Zoom et épaisseur : effet immédiat. Tolérance, raccord et couleurs :
     recalcul automatique de l'aperçu — état : <span id="status">prêt</span></p>
  <div id="apercu" class="apercu">{{ svg_content|safe }}</div>
  <table>
    <tr><th>Contraste (écart-type)</th><td>{{ report.analysis.contrast_std }}</td></tr>
    <tr><th>Bruit estimé</th><td>{{ report.analysis.noise_sigma }}</td></tr>
    <tr><th>Densité du tracé</th><td>{{ report.analysis.ink_fraction }}</td></tr>
    <tr><th>Épaisseur de trait estimée</th><td>{{ report.analysis.stroke_width_px }} px</td></tr>
  </table>
  <a class="btn" href="{{ url_for('download', job_id=job_id, filename=svg_name, dl=1) }}">Télécharger le SVG</a>
  <a class="btn" href="{{ url_for('download', job_id=job_id, filename=dxf_name, dl=1) }}">Télécharger le DXF</a>
  <a class="btn secondary" href="{{ url_for('download', job_id=job_id, filename=report_name, dl=1) }}">Rapport JSON</a>
  <p style="font-size:.85rem;color:#666">Les fichiers téléchargés correspondent à la conversion initiale.
     Après ajustement des curseurs, régénérez-les :</p>
  <button type="button" id="regen">Régénérer les fichiers avec ces réglages</button>
  <br><a class="back" href="{{ url_for('index') }}">← Convertir une autre image</a>
</div>
<script>
(function () {
  var apercu = document.getElementById('apercu');
  var svg0 = apercu.querySelector('svg');
  var natW = svg0 ? parseFloat(svg0.getAttribute('width')) || 800 : 800;
  var gen = 0, timer = null;
  var base = {{ url_for('api_preview', job_id=job_id)|tojson }};

  function applyZoom() {
    var z = document.getElementById('zoom').value;
    var s = apercu.querySelector('svg');
    if (s) { s.style.width = (natW * z / 100) + 'px'; s.style.height = 'auto'; }
    document.getElementById('zoom_val').textContent = z;
  }
  function applyStroke() {
    var v = document.getElementById('sw').value;
    var s = apercu.querySelector('svg');
    if (s) {
      var g = s.querySelector('g');
      if (g) g.setAttribute('stroke-width', v);
    }
    document.getElementById('sw_val').textContent = v;
  }
  function schedule() {
    document.getElementById('status').textContent = 'recalcul…';
    clearTimeout(timer);
    timer = setTimeout(recompute, 700);
  }
  async function recompute() {
    var my = ++gen;
    var q = new URLSearchParams({
      tolerance: document.getElementById('tol').value,
      gap: document.getElementById('gap').value,
      colors: document.getElementById('live_colors').value || 'none'
    });
    try {
      var r = await fetch(base + '?' + q.toString());
      var d = await r.json();
      if (my !== gen) return;
      if (d.error) { document.getElementById('status').textContent = 'erreur : ' + d.error; return; }
      apercu.innerHTML = d.svg;
      applyZoom(); applyStroke();
      document.getElementById('stats').innerHTML = '<strong>{{ report.source|basename }}</strong> — ' +
        d.paths + ' tracés (dont ' + d.closed + ' fermés, ' + d.circles + ' cercles, ' +
        d.ellipses + ' ellipses et ' + d.arcs + ' arcs parfaits), ' + d.points + ' points.';
      document.getElementById('status').textContent = 'prêt';
    } catch (e) {
      if (my === gen) document.getElementById('status').textContent = 'erreur réseau';
    }
  }
  document.getElementById('zoom').addEventListener('input', applyZoom);
  document.getElementById('sw').addEventListener('input', applyStroke);
  ['tol', 'gap'].forEach(function (id) {
    document.getElementById(id).addEventListener('input', function () {
      document.getElementById(id.replace('tol', 'tol_val').replace('gap', 'gap_val')).textContent =
        document.getElementById(id).value;
      schedule();
    });
  });
  document.getElementById('live_colors').addEventListener('change', schedule);
  document.getElementById('regen').addEventListener('click', async function () {
    var b = this;
    b.disabled = true; b.textContent = 'Régénération…';
    try {
      var r = await fetch({{ url_for('regenerate', job_id=job_id)|tojson }}, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          tolerance: document.getElementById('tol').value,
          gap: document.getElementById('gap').value,
          stroke_width: document.getElementById('sw').value,
          colors: document.getElementById('live_colors').value || 'none'
        })
      });
      if (r.ok) { location.reload(); return; }
      var d = await r.json().catch(function () { return {}; });
      alert('Régénération impossible : ' + (d.error || r.status));
    } catch (e) {
      alert('Erreur réseau');
    }
    b.disabled = false; b.textContent = 'Régénérer les fichiers avec ces réglages';
  });
  applyZoom(); applyStroke();
})();
</script>"""


@app.template_filter("basename")
def basename(value: str) -> str:
    return Path(value).name


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HEAD + FORM_BODY + FOOT)


@app.route("/convert", methods=["POST"])
def convert():
    upload = request.files.get("image")
    error = None
    if upload is None or not upload.filename:
        error = "Aucun fichier reçu."
    else:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            error = "Format non pris en charge. Formats acceptés : JPEG, PNG, BMP, WebP, TIFF."
    if error:
        return render_template_string(HEAD + FORM_BODY + FOOT, error=error), 400

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload.filename).stem)[:60] or "image"
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    src = job_dir / f"{stem}{suffix}"
    upload.save(src)

    try:
        scale = float(request.form.get("scale", "1.0").replace(",", "."))
        if scale <= 0:
            raise ValueError
    except ValueError:
        scale = 1.0

    close_lines = request.form.get("close") == "1"
    raw_stroke = request.form.get("stroke_width", "").strip().lower()
    stroke_width = None
    if raw_stroke and raw_stroke != "auto":
        try:
            stroke_width = float(raw_stroke.replace(",", "."))
            if stroke_width <= 0:
                raise ValueError
        except ValueError:
            stroke_width = None
    agent = ImageVectorAgent(
        mode=request.form.get("mode", "auto"),
        quality=request.form.get("quality", "high"),
        unit=request.form.get("unit", "px"),
        scale=scale,
        invert=request.form.get("invert") == "1",
        close_lines=close_lines,
        closed_only=False,
        stroke_width=stroke_width,
        fit_curves=request.form.get("curves") == "1",
        colors=request.form.get("colors", "none"),
    )
    try:
        report = agent.run(src, job_dir)
    except Exception as e:
        return render_template_string(HEAD + FORM_BODY + FOOT, error=f"Conversion impossible : {e}"), 400
    return redirect(url_for("result", job_id=job_id))


def _load_job(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    job_dir = JOBS_DIR / job_id
    reports = sorted(job_dir.glob("*_report.json"))
    if not reports:
        abort(404)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    src = Path(report["source"])
    if not src.is_file():
        abort(404)
    return job_dir, report, src


def _parse_float(value: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _colors_str(report: dict) -> str:
    c = report.get("parameters", {}).get("colors", "none")
    if isinstance(c, list):
        return ", ".join(c)
    return "" if c in ("none", None) else str(c)


@app.route("/result/<job_id>")
def result(job_id: str):
    job_dir, report, _src = _load_job(job_id)
    params = report.get("parameters", {})
    w = report["analysis"]["width"]
    h = report["analysis"]["height"]
    if params.get("close_lines"):
        gap0 = params.get("gap_px")
        init_gap = gap0 if gap0 is not None else max(1.5, 0.004 * max(w, h))
    else:
        init_gap = 0.0
    init_tol = min(3.0, max(0.45, params.get("tolerance_px", 0.8)))
    svg_content = Path(report["outputs"]["svg"]).read_text(encoding="utf-8")
    return render_template_string(
        HEAD + RESULT_BODY + FOOT,
        job_id=job_id,
        report=report,
        svg_content=svg_content,
        init_sw=report["analysis"]["stroke_width_px"],
        init_tol=round(init_tol, 2),
        init_gap=round(init_gap, 2),
        init_colors=_colors_str(report),
        svg_name=Path(report["outputs"]["svg"]).name,
        dxf_name=Path(report["outputs"]["dxf"]).name,
        preview_name=Path(report["outputs"]["preview"]).name,
        report_name=sorted(job_dir.glob("*_report.json"))[0].name,
    )


@app.route("/api/preview/<job_id>")
def api_preview(job_id: str):
    _job_dir, report, src = _load_job(job_id)
    tol = _parse_float(request.args.get("tolerance", "0.8"), 0.8, 0.3, 4.0)
    gap = _parse_float(request.args.get("gap", "0"), 0.0, 0.0, 20.0)
    colors = request.args.get("colors", "none") or "none"
    live = JOBS_DIR / job_id / "live"
    try:
        agent = ImageVectorAgent(quality="high", tolerance=tol,
                                 close_lines=gap > 0, gap=gap if gap > 0 else None,
                                 colors=colors)
        rep = agent.run(src, live)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    svg = (live / (src.stem + ".svg")).read_text(encoding="utf-8")
    return jsonify({"svg": svg, "paths": rep["paths"], "closed": rep["closed_paths"],
                    "circles": rep["circles"], "arcs": rep["arcs"],
                    "ellipses": rep["ellipses"], "points": rep["points"]})


@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate(job_id: str):
    _job_dir, report, src = _load_job(job_id)
    params = report.get("parameters", {})
    body = request.get_json(silent=True) or {}
    tol = _parse_float(body.get("tolerance", params.get("tolerance_px", 0.8)), 0.8, 0.3, 4.0)
    gap = _parse_float(body.get("gap", "0"), 0.0, 0.0, 20.0)
    sw = _parse_float(body.get("stroke_width", report["analysis"]["stroke_width_px"]),
                      report["analysis"]["stroke_width_px"], 0.25, 6.0)
    colors = body.get("colors", "none") or "none"
    try:
        agent = ImageVectorAgent(
            mode=params.get("mode", "auto"),
            tolerance=tol,
            threshold=params.get("threshold", "auto"),
            invert=bool(params.get("invert", False)),
            unit=params.get("unit", "px"),
            scale=params.get("scale", 1.0),
            close_lines=gap > 0,
            gap=gap if gap > 0 else None,
            stroke_width=sw,
            colors=colors,
        )
        agent.run(src, JOBS_DIR / job_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route("/download/<job_id>/<path:filename>")
def download(job_id: str, filename: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    job_dir = (JOBS_DIR / job_id).resolve()
    target = (job_dir / Path(filename).name).resolve()
    if target.parent != job_dir or not target.is_file():
        abort(404)
    return send_file(target, as_attachment=request.args.get("dl") == "1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interface web de l'agent analyste d'image")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    JOBS_DIR.mkdir(exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
