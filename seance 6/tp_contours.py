# -*- coding: utf-8 -*-
# Script complet exporté du notebook. Exécuter depuis le dossier du TP.


# %pip install "opencv-python-headless==4.12.0.88" numpy matplotlib ipython scikit-image

from pathlib import Path
from time import perf_counter
import sys, platform, hashlib, csv
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
IMAGE_PATH = ROOT / "image.jpg"
PROTO_PATH = ROOT / "deploy.prototxt"
MODEL_PATH = ROOT / "hed_pretrained_bsds.caffemodel"
OUT = ROOT / "resultats"
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

for path in (IMAGE_PATH, PROTO_PATH, MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier requis absent : {path}. Voir les sources en fin de notebook.")
print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
print(f"Matplotlib {matplotlib.__version__} | {platform.system()} {platform.machine()}")
print(f"CPU : OpenCV limité à {cv2.getNumThreads()} threads")
print("SHA-256 du modèle :", hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())

def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

def show_grid(images, titles, filename, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    for ax, im, title in zip(axes.flat, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def save_map(name, a):
    ok = cv2.imwrite(str(OUT / name), np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8))
    if not ok:
        raise OSError(f"Échec d'enregistrement : {name}")

original_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
if original_bgr is None:
    raise ValueError("L'image ne peut pas être décodée.")
h0, w0 = original_bgr.shape[:2]
scale = min(1.0, 384 / max(h0, w0))
img = cv2.resize(original_bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
print(f"Original : {w0} × {h0} ; résolution de travail : {w} × {h}")
cv2.imwrite(str(OUT / "image_travail.png"), img)
show_grid([rgb, gray / 255., blur / 255.],
          ["Image couleur", "Niveaux de gris", "Flou gaussien (5 × 5, σ = 1,2)"], "A_pretraitement.png")

def sobel_map(b):
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    return normalize01(cv2.magnitude(gx, gy))

sobel = sobel_map(blur)
canny_low = cv2.Canny(blur, 50, 150)
canny = cv2.Canny(blur, 100, 200)
canny_norm = canny.astype(np.float32) / 255.0
show_grid([rgb, sobel, canny_low / 255., canny_norm],
          ["Image originale (résolution de travail)", "Sobel : norme du gradient",
           "Canny : seuils (50, 150)", "Canny : seuils (100, 200)"], "A_contours.png", cols=2)
n_low, n_high = np.count_nonzero(canny_low), np.count_nonzero(canny)
display(Markdown(f"**Mesure :** {n_low:,} pixels de contour avec (50, 150), contre {n_high:,} avec (100, 200), soit une diminution de **{100*(n_low-n_high)/max(n_low,1):.1f} %**. Cette quantité mesure la densité, pas la justesse des contours."))
save_map("resultat_sobel.png", sobel)
save_map("resultat_canny_50_150.png", canny_low / 255.)
save_map("resultat_canny.png", canny_norm)

class CropLayer:
    def __init__(self, params, blobs):
        self.top = self.left = self.bottom = self.right = 0

    def getMemoryShapes(self, inputs):
        source, target = inputs
        height, width = target[2:4]
        self.top = (source[2] - height) // 2
        self.left = (source[3] - width) // 2
        self.bottom, self.right = self.top + height, self.left + width
        return [[source[0], source[1], height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.top:self.bottom, self.left:self.right]]

# Autorise la réexécution de la cellule dans un même noyau.
try:
    cv2.dnn_unregisterLayer("Crop")
except cv2.error:
    pass
cv2.dnn_registerLayer("Crop", CropLayer)
t0 = perf_counter()
net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
load_ms = (perf_counter() - t0) * 1000
print(f"Chargement du modèle : {load_ms:.1f} ms (séparé de l'inférence)")

def predict_hed(bgr):
    height, width = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        bgr, scalefactor=1.0, size=(width, height),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
    net.setInput(blob)
    raw = net.forward()[0, 0]
    raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    return raw, normalize01(raw)

hed_raw, hed = predict_hed(img)
print(f"Sortie brute : [{hed_raw.min():.4f}, {hed_raw.max():.4f}]")
print(f"Après normalisation : [{hed.min():.4f}, {hed.max():.4f}]")
show_grid([rgb, canny_norm, hed], ["Image étudiée", "Canny (100, 200)", "HED : carte continue"], "B_HED.png")
save_map("resultat_hed.png", hed)

alphas = [0.3, 0.6, 0.8]
fusions = {a: a * hed + (1 - a) * canny_norm for a in alphas}
for a, f in fusions.items():
    assert f.shape == hed.shape == canny_norm.shape
    assert np.isfinite(f).all() and f.min() >= 0 and f.max() <= 1 + 1e-6
    save_map(f"fusion_{a:.1f}.png", f)
show_grid(list(fusions.values()), [f"Fusion α = {a:.1f}" for a in alphas], "C_fusions.png")

# Deux régions : tête/visage, puis combinaison/inscriptions.
regions = {"Tête et visage": (0.12, 0.46, 0.32, 0.60),
           "Combinaison et détails": (0.52, 0.91, 0.12, 0.60)}
for label, (y1, y2, x1, x2) in regions.items():
    ys, xs = slice(round(y1*h), round(y2*h)), slice(round(x1*w), round(x2*w))
    show_grid([rgb[ys, xs], canny_norm[ys, xs], hed[ys, xs],
               fusions[0.3][ys, xs], fusions[0.6][ys, xs], fusions[0.8][ys, xs]],
              [label, "Canny", "HED", "α = 0,3", "α = 0,6", "α = 0,8"],
              "C_zoom_visage.png" if "visage" in label else "C_zoom_combinaison.png")

def classical_input(bgr):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)

def run_canny():
    return cv2.Canny(classical_input(img), 100, 200).astype(np.float32) / 255.

def run_fusion():
    c = run_canny()
    _, hh = predict_hed(img)
    return 0.6 * hh + 0.4 * c

def benchmark(fn, repeats=5):
    fn()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        durations.append(1000 * (perf_counter() - start))
    return float(np.mean(durations)), float(np.std(durations, ddof=1))

operations = {
    "Sobel": lambda: sobel_map(classical_input(img)),
    "Canny (100, 200)": run_canny,
    "HED": lambda: predict_hed(img),
    "Fusion α=0,6 (chaîne complète)": run_fusion,
    "Fusion seule (cartes disponibles)": lambda: 0.6 * hed + 0.4 * canny_norm,
}
timings = {name: benchmark(fn) for name, fn in operations.items()}
rows = ["| Méthode | Moyenne (ms) | Écart-type (ms) |", "|:--|--:|--:|"]
rows += [f"| {name} | {mean:.3f} | {std:.3f} |" for name, (mean, std) in timings.items()]
display(Markdown("\n".join(rows)))
with (OUT / "temps_execution.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["methode", "moyenne_ms", "ecart_type_ms"])
    writer.writerows((name, *values) for name, values in timings.items())
display(Markdown(f"Dans cette exécution, HED prend environ **{timings['HED'][0]/timings['Canny (100, 200)'][0]:.0f} fois** le temps de Canny. Cinq répétitions donnent une indication locale ; la charge de la machine et la taille de l'image influencent ces valeurs."))

comparison = [
    ("Sobel", "Bandes larges ; pas de chaînage explicite", "Fortes réponses aux variations locales ; bruit atténué par le flou", "Sobel", "Visualise la force du gradient, sans binarisation"),
    ("Canny (100, 200)", "Lignes fines ; ruptures possibles aux faibles contrastes", "Détails contrastés conservés ; seuils et flou réduisent les réponses faibles", "Canny (100, 200)", "Rapide ; compromis précision spatiale / perte de détails"),
    ("HED", "Grandes structures plus larges et progressives", "Atténuation de certains détails ; dépend du domaine appris", "HED", "Carte continue ; contours parfois épais ; pas d'immunité au bruit"),
    ("Fusion α=0,6", "Superposition de lignes fines et bandes HED", "Peut réintroduire les détails et défauts des deux cartes", "Fusion α=0,6 (chaîne complète)", "Compromis visuel ; coût dominé par HED"),
]
header = "| Méthode | Continuité des contours | Sensibilité aux textures/bruit | Coût moyen CPU | Observations |\n|:--|:--|:--|--:|:--|"
body = [f"| {name} | {continuity} | {noise} | {timings[key][0]:.3f} ms | {obs} |" for name, continuity, noise, key, obs in comparison]
display(Markdown(header + "\n" + "\n".join(body)))
(OUT / "tableau_comparatif.md").write_text(header + "\n" + "\n".join(body), encoding="utf-8")

maps = {"Sobel": sobel, "Canny (50, 150)": canny_low / 255.,
        "Canny (100, 200)": canny_norm, "HED": hed,
        **{f"Fusion α={a:.1f}": f for a, f in fusions.items()}}
measures = []
for name, a in maps.items():
    mask = (a >= 0.5).astype(np.uint8)
    nlabels, _ = cv2.connectedComponents(mask, connectivity=8)
    measures.append((name, int(mask.sum()), 100 * float(mask.mean()), nlabels - 1))
lines = ["| Carte (seuil 0,5) | Pixels actifs | Densité (%) | Composantes |", "|:--|--:|--:|--:|"]
lines += [f"| {name} | {n} | {density:.2f} | {components} |" for name, n, density, components in measures]
display(Markdown("\n".join(lines)))
with (OUT / "mesures_cartes.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["carte", "pixels_actifs", "densite_pourcent", "composantes_8_connexes"])
    writer.writerows(measures)

def contour_metrics(prediction, reference, threshold=0.5):
    """prediction : scores dans [0,1] ; reference : masque booléen aligné."""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError("Les deux cartes doivent être 2D et de même taille.")
    if reference.dtype != np.bool_:
        raise TypeError("Convertir explicitement la vérité terrain en masque booléen.")
    if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("Les scores doivent être finis et dans [0,1].")
    pred, ref = prediction >= threshold, reference
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    def ratio(a, b):
        return a / b if b else float("nan")  # Indéfini si dénominateur nul.
    return {"VP": tp, "FP": fp, "FN": fn,
            "precision": ratio(tp, tp + fp), "rappel": ratio(tp, tp + fn),
            "F1": ratio(2 * tp, 2 * tp + fp + fn), "IoU": ratio(tp, tp + fp + fn)}

# Exemple à activer uniquement avec une annotation réelle :
# gt_image = cv2.imread("verite_terrain.png", cv2.IMREAD_GRAYSCALE)
# gt = gt_image > 0  # À adapter au codage réel de l'annotation.
# print(contour_metrics(hed, gt, threshold=0.5))
print("Aucune vérité terrain fournie : aucun score de précision, rappel ou F1 annoncé.")

hed_binary = hed >= 0.5
union = np.logical_or(hed_binary, canny_norm > 0)
assert np.all(union[hed_binary]) and np.all(union[canny_norm > 0])
show_grid([canny_norm, hed_binary.astype(float), union.astype(float)],
          ["Canny (100, 200)", "HED ≥ 0,5", "Union logique"], "extension_union.png")
extra = np.count_nonzero(union & ~(canny_norm > 0))
display(Markdown(f"L'union contient **{np.count_nonzero(union):,} pixels**, contre **{np.count_nonzero(canny):,}** pour Canny et **{np.count_nonzero(hed_binary):,}** pour HED seuillé. Elle ajoute **{extra:,} pixels** à Canny ; cet ajout ne démontre pas un gain de précision."))
save_map("fusion_logique.png", union.astype(float))
