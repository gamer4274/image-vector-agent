# Agent analyste d'image → SVG + DXF

Convertit une image JPEG (scan de plan, dessin technique, schéma, logo) en fichiers vectoriels de haute fidélité :

- **SVG** : tracé pur sans fioriture — uniquement des `<path>` en `fill="none"`, un seul groupe, coordonnées fidèles à l'image d'origine.
- **DXF** (R2010) : polylignes `LWPOLYLINE` sur calque `TRACE`, unités configurables, compatible AutoCAD/LibreCAD/QCAD.
- **Rapport JSON** : analyse de l'image et paramètres utilisés.
- **Aperçu PNG** : rendu du tracé pour contrôle visuel.

## Fonctionnement de l'agent

1. **Analyse** : contraste, netteté (variance du Laplacien), bruit estimé, polarité du tracé (sombre/clair).
2. **Prétraitement** : sur-échantillonnage jusqu'à 1600 px si l'image est petite, débruitage adaptatif (fastNlMeans), rehaussement CLAHE si faible contraste.
3. **Binarisation** : Otsu si l'histogramme est nettement bimodal, sinon seuillage adaptatif gaussien (photos, scans irréguliers). Suppression des composantes parasites.
4. **Extraction** :
   - mode `centerline` (défaut) : squelettisation puis parcours du graphe de pixels → tracés centraux d'un seul trait, idéaux pour plans et dessins au trait ;
   - mode `outline` : contours des zones pleines (logos, formes remplies) ;
   - mode `auto` : choisit selon la densité d'encre.
5. **Simplification** : Douglas-Peucker avec tolérance sub-pixel (préréglage `high` = 0,8 px) pour un fichier léger mais fidèle.
6. **Export** SVG + DXF + aperçu + rapport.

## Usage

```bash
pip install -r requirements.txt

python agent.py image.jpg -o sortie/
python agent.py scan.jpg -o sortie/ --quality max --unit mm --scale 0.1
python agent.py logo.jpg -o sortie/ --mode outline
python agent.py plan.jpg -o sortie/ --threshold adaptive --min-area 1e-5
```

| Option | Valeurs | Rôle |
|---|---|---|
| `--mode` | `auto`, `centerline`, `outline` | Stratégie d'extraction |
| `--quality` | `max`, `high`, `balanced` | Tolérance de simplification (0,45 / 0,8 / 1,6 px) |
| `--tolerance` | float | Tolérance manuelle en px image |
| `--threshold` | `auto`, `otsu`, `adaptive` | Méthode de binarisation |
| `--invert` | flag | Tracé clair sur fond sombre |
| `--unit` | `px`, `mm`, `cm`, `in` | Unités du DXF |
| `--scale` | float | Échelle des coordonnées DXF (ex. 0,1 = 1 px → 0,1 mm) |
| `--min-area` | float | Seuil anti-parasites (fraction de surface, défaut 2e-5) |

## Qualité

- Le SVG conserve les dimensions exactes de l'image (`viewBox` = taille d'origine).
- Le mode `centerline` produit un trait unique par trait de crayon (pas de double contour).
- `--quality max` réduit la simplification au minimum (0,45 px) au prix de plus de points.
- L'épaisseur de trait SVG est estimée automatiquement depuis la transformée de distance du masque.
