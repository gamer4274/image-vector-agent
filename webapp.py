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

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

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
  <p><strong>{{ report.source }}</strong> — {{ report.analysis.width }}×{{ report.analysis.height }} px,
     mode {{ report.analysis.mode_used }}, seuillage {{ report.analysis.threshold_method }},
     {{ report.paths }} tracés (dont {{ report.closed_paths }} fermés), {{ report.points }} points.</p>
  <img class="preview" src="{{ url_for('download', job_id=job_id, filename=preview_name) }}" alt="Aperçu du tracé">
  <table>
    <tr><th>Contraste (écart-type)</th><td>{{ report.analysis.contrast_std }}</td></tr>
    <tr><th>Bruit estimé</th><td>{{ report.analysis.noise_sigma }}</td></tr>
    <tr><th>Densité du tracé</th><td>{{ report.analysis.ink_fraction }}</td></tr>
    <tr><th>Épaisseur de trait estimée</th><td>{{ report.analysis.stroke_width_px }} px</td></tr>
  </table>
  <a class="btn" href="{{ url_for('download', job_id=job_id, filename=svg_name, dl=1) }}">Télécharger le SVG</a>
  <a class="btn" href="{{ url_for('download', job_id=job_id, filename=dxf_name, dl=1) }}">Télécharger le DXF</a>
  <a class="btn secondary" href="{{ url_for('download', job_id=job_id, filename=report_name, dl=1) }}">Rapport JSON</a>
  <br><a class="back" href="{{ url_for('index') }}">← Convertir une autre image</a>
</div>"""


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
    )
    try:
        report = agent.run(src, job_dir)
    except Exception as e:
        return render_template_string(HEAD + FORM_BODY + FOOT, error=f"Conversion impossible : {e}"), 400
    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    job_dir = JOBS_DIR / job_id
    reports = sorted(job_dir.glob("*_report.json"))
    if not reports:
        abort(404)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    return render_template_string(
        HEAD + RESULT_BODY + FOOT,
        job_id=job_id,
        report=report,
        svg_name=Path(report["outputs"]["svg"]).name,
        dxf_name=Path(report["outputs"]["dxf"]).name,
        preview_name=Path(report["outputs"]["preview"]).name,
        report_name=reports[0].name,
    )


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
