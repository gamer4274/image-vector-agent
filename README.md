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
   - mode `edges` : détection d'arêtes (Canny) puis tracé central — capture tous les détails internes d'une image tonale (cercles, engrenages, visages) ;
   - mode `auto` : choisit selon la densité d'encre, détecte un fond sombre (polarité inversée automatique) et bascule vers `edges` si le tracé central est trop bruité.
5. **Simplification** : Douglas-Peucker avec tolérance sub-pixel (préréglage `high` = 0,8 px) pour un fichier léger mais fidèle.
6. **Export** SVG + DXF + aperçu + rapport.

## Usage

### Interface web

```bash
pip install -r requirements.txt
python webapp.py
```

Ouvrez ensuite http://127.0.0.1:8000 dans un navigateur : chargez une image, choisissez
les options, puis téléchargez le SVG et le DXF depuis la page de résultat (aperçu inclus).

Options disponibles : mode d'extraction, qualité du tracé, unités et échelle du DXF,
tracé clair sur fond sombre. Formats acceptés : JPEG, PNG, BMP, WebP, TIFF (32 Mo max).

### Ligne de commande

```bash
pip install -r requirements.txt

python agent.py image.jpg -o sortie/
python agent.py scan.jpg -o sortie/ --quality max --unit mm --scale 0.1
python agent.py logo.jpg -o sortie/ --mode outline
python agent.py plan.jpg -o sortie/ --threshold adaptive --min-area 1e-5
```

| Option | Valeurs | Rôle |
|---|---|---|
| `--mode` | `auto`, `centerline`, `outline`, `edges` | Stratégie d'extraction |
| `--quality` | `max`, `high`, `balanced` | Tolérance de simplification (0,45 / 0,8 / 1,6 px) |
| `--tolerance` | float | Tolérance manuelle en px image |
| `--threshold` | `auto`, `otsu`, `adaptive` | Méthode de binarisation |
| `--invert` | flag | Tracé clair sur fond sombre |
| `--unit` | `px`, `mm`, `cm`, `in` | Unités du DXF |
| `--scale` | float | Échelle des coordonnées DXF (ex. 0,1 = 1 px → 0,1 mm) |
| `--min-area` | float | Seuil anti-parasites (fraction de surface, défaut 2e-5) |
| `--close` | flag | Ferme les tracés : raccorde les extrémités proches en boucles (gravure/découpe laser) |
| `--closed-only` | flag | Ne conserve que les tracés fermés |
| `--gap` | float | Distance max de raccord des extrémités en px (défaut auto : max(1,5 px ; 0,4 % de la taille)) |

## Gravure laser

Pour préparer un DXF à la gravure ou la découpe laser, les tracés doivent être fermés :

```bash
python agent.py logo.jpg -o sortie/ --mode edges --close --unit mm --scale 0.1
```

- `--close` raccorde les extrémités proches en plusieurs passes (distance croissante), fusionne les segments en chaînes et ferme les boucles. Aucune géométrie n'est supprimée : les traits intrinsèquement ouverts (qui ne peuvent être refermés sans tracer une fausse ligne) restent ouverts ;
- `--closed-only` (découpe) : supprime en plus les traits restés ouverts pour un DXF 100 % contours fermés — à réserver à la découpe, car cela retire du contenu ;
- ajustez `--gap` si des boucles ne se ferment pas (augmenter) ou si des zones distinctes se recollent (diminuer).

Dans l'interface web, la case « Tracés fermés (gravure / découpe laser) » active `--close` sans suppression.

## Qualité

- Le SVG conserve les dimensions exactes de l'image (`viewBox` = taille d'origine).
- Le mode `centerline` produit un trait unique par trait de crayon (pas de double contour).
- `--quality max` réduit la simplification au minimum (0,45 px) au prix de plus de points.
- L'épaisseur de trait SVG est estimée automatiquement depuis la transformée de distance du masque.
