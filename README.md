# Agent analyste d'image → SVG + DXF

Convertit une image JPEG (scan de plan, dessin technique, schéma, logo) en fichiers vectoriels de haute fidélité :

- **SVG** : tracé pur sans fioriture — `<path>` en `fill="none"` plus les entités exactes reconnues (`<circle>`, `<ellipse>`), un seul groupe, coordonnées fidèles à l'image d'origine.
- **DXF** (R2010) : polylignes `LWPOLYLINE` sur calque `TRACE` plus les entités exactes `CIRCLE`/`ARC`/`ELLIPSE`, unités configurables, compatible AutoCAD/LibreCAD/QCAD.
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
6. **Cercles, arcs et ellipses parfaits (CAO)** : chaque tracé dont les points s'ajustent à une géométrie circulaire (erreur radiale faible, couverture angulaire continue) devient une entité exacte — `CIRCLE`/`ARC` dans le DXF, `<circle>`/arc SVG — avec une précision typique de l'ordre du dixième de pixel. Les « faux cercles » (ellipses et ovales vus de biais) sont détectés par un passage dédié (`cv2.fitEllipse` sur les contours, affinement borné, contrôle d'épaisseur de trait) et deviennent des entités `ELLIPSE` DXF / `<ellipse>` SVG exactes (centre, demi-axes, inclinaison). Désactivable avec `--no-curves`.
7. **Anneaux concentriques (détection dédiée)** : les grands cercles réguliers d'une photo (cadre d'un médaillon, bord d'une pièce, roue) ne suivent pas toujours les tracés extraits. Un second passage combine un balayage radial autour des principaux centres de l'image (anneaux concentriques, y compris à double piste et multi-centres) et une transformée de Hough (petits cercles, multi-centres, sans plafond arbitraire). Chaque candidat est affiné puis validé par des tests de couverture d'arêtes, d'isolement radial (proéminence), de contraste et de **dispersion radiale** — ce dernier rejette les faux positifs circiformes (lettres « O » et glyphes de texte). Limites connues sur photos très texturées : les petits cercles noyés dans la texture (rivets) et les cercles internes des engrenages à double piste serrée peuvent ne pas être promus en entités `CIRCLE` exactes ; ils restent alors dans le tracé général (polylignes).
8. **Export** SVG + DXF + aperçu + rapport.

## Usage

### Interface web

```bash
pip install -r requirements.txt
python webapp.py
```

Ouvrez ensuite http://127.0.0.1:8000 dans un navigateur : chargez une image, choisissez
les options, puis téléchargez le SVG et le DXF depuis la page de résultat.

La page de résultat affiche un **aperçu interactif en direct** avec curseurs :

- **Zoom** et **épaisseur du trait** : effet immédiat (côté navigateur) ;
- **Tolérance de simplification**, **raccord des extrémités** (0 = pas de fermeture)
  et **couleurs à tracer** : l'aperçu est recalculé automatiquement (léger délai) ;
- le bouton « Régénérer les fichiers avec ces réglages » réécrit le SVG et le DXF
  téléchargeables avec les réglages courants.

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
| `--stroke-width` | float | Épaisseur du trait SVG en px (défaut : estimation automatique) |
| `--no-curves` | flag | Désactive la détection de cercles/arcs parfaits |
| `--colors` | liste ou `toutes` | Trace en couleurs : quantification en 6 catégories (bleu, blanc, rouge, vert, jaune, noir), séparées par des virgules (défaut : aucune) |

## Gravure laser

Pour préparer un DXF à la gravure ou la découpe laser, les tracés doivent être fermés :

```bash
python agent.py logo.jpg -o sortie/ --mode edges --close --unit mm --scale 0.1
```

- `--close` raccorde les extrémités proches en plusieurs passes (distance croissante), fusionne les segments en chaînes et ferme les boucles. Aucune géométrie n'est supprimée : les traits intrinsèquement ouverts (qui ne peuvent être refermés sans tracer une fausse ligne) restent ouverts ;
- `--closed-only` (découpe) : supprime en plus les traits restés ouverts pour un DXF 100 % contours fermés — à réserver à la découpe, car cela retire du contenu ;
- ajustez `--gap` si des boucles ne se ferment pas (augmenter) ou si des zones distinctes se recollent (diminuer).

Dans l'interface web, la case « Tracés fermés (gravure / découpe laser) » active `--close` sans suppression.

## Couleurs

Par défaut le tracé est monochrome (toutes les formes en noir). L'option `--colors` active
le tracé en couleurs : l'image est d'abord quantifiée en 6 catégories simples
(bleu, blanc, rouge, vert, jaune, noir) selon des règles HSV (luminosité, saturation,
teinte), puis chaque catégorie est tracée séparément — les dégradés sont donc aplatis
vers leur catégorie dominante, ce qui simplifie le tracé avant extraction.

```bash
python agent.py logo.jpg -o sortie/ --colors rouge,jaune
python agent.py logo.jpg -o sortie/ --colors toutes
```

- sélection des couleurs à traiter : une catégorie absente ou quasi absente (< 0,3 % de la surface) est ignorée ; une erreur liste les choix valides si le nom est inconnu ;
- les cercles, arcs et ellipses détectés prennent la couleur majoritaire de leur emplacement ;
- le SVG utilise une couleur de trait par forme (fond clair `#fafafa`, blanc rendu `#e0e0e0` pour rester visible) ;
- le DXF range chaque couleur dans un calque dédié `COL_BLEU`, `COL_ROUGE`, … avec la couleur AutoCAD correspondante ;
- dans l'interface web, le champ « Couleurs à tracer » accepte la même syntaxe (vide = monochrome).

## Qualité

- Le SVG conserve les dimensions exactes de l'image (`viewBox` = taille d'origine).
- Le mode `centerline` produit un trait unique par trait de crayon (pas de double contour).
- `--quality max` réduit la simplification au minimum (0,45 px) au prix de plus de points.
- L'épaisseur de trait SVG est estimée automatiquement depuis la transformée de distance du masque, ou fixée manuellement avec `--stroke-width` (ex. `--stroke-width 1` pour un trait fin).

## Tests

```bash
python -m unittest test_agent -v
```

Couvre : ajustement de cercles, scores d'anneaux, détection de cercles parfaits (complets, occultés, concentriques à double piste, petits cercles), détection d'ellipses (complètes, inclinées, occultées) et rejet des faux positifs (quasi-cercles, cercles, lignes droites, pistes/puces), borne de temps de `detect_rings`, détection d'arcs, fermeture de tracés (gravure laser), exports SVG/DXF (`CIRCLE`/`ARC`/`ELLIPSE`/`LWPOLYLINE`) et un test bout-en-bout du pipeline.
