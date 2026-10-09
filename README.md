# Vision par Ordinateur - Travaux Pratiques et Algorithmes

**Auteur :** Amine BOURAMTANE  
**Domaine :** Traitement d'Image et Vision par Ordinateur  
**Technologies :** Python 3, OpenCV, NumPy, Matplotlib, Pandas, Jupyter Notebook  

---

## Table des Matières

1. Présentation du Projet
2. Prérequis et Environnement de Développement
3. Structure du Répertoire
4. Contenu des Séances
   - Séance 1 : Prise en Main et Mesures Géométriques
   - Séance 2 : Opérations Élémentaires et Amélioration d'Histogrammes
   - Séance 3 : Histogrammes Avancés et Seuillage
5. Concepts Théoriques et Algorithmes Implémentés
6. Instructions d'Exécution
7. Auteur et Contact

---

## 1. Présentation du Projet

Ce dépôt rassemble l'ensemble des travaux pratiques (TP), des notebooks d'expérimentation et des supports théoriques relatifs au module de **Vision par Ordinateur**.

L'objectif principal est d'explorer et d'implémenter les algorithmes fondamentaux du traitement d'images numériques, allant de la manipulation de base des matrices de pixels jusqu'à la segmentation par seuillage automatique et la caractérisation géométrique d'objets.

---

## 2. Prérequis et Environnement de Développement

### Dépendances requises

Le projet s'appuie sur la pile scientifique standard Python 3.x :
- **Python** (version 3.8 ou supérieure)
- **OpenCV** (`opencv-python`) : Traitement d'images et vision par ordinateur.
- **NumPy** : Calcul matriciel et manipulation de tableaux multidimensionnels.
- **Matplotlib** : Visualisation d'images, de canaux de couleurs et d'histogrammes.
- **Pandas** : Analyse et manipulation des données tabulaires de mesures d'objets.
- **Jupyter Notebook / JupyterLab** : Environnement d'exécution interactif.

### Installation de l'environnement

```bash
# Clone du dépôt
git clone https://github.com/Aminebouramtane/vision-par-ordinateur.git
cd vision-par-ordinateur

# Création et activation d'un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate  # Sur Linux/macOS
# venv\Scripts\activate   # Sur Windows

# Installation des paquets requis
pip install opencv-python numpy matplotlib pandas jupyter
```

---

## 3. Structure du Répertoire

```
vision-par-ordinateur/
│
├── seance 1/
│   ├── TP0_Prise_en_main_Vision_Ordinateur.ipynb  # TP0 : Manipulation d'images, espaces BGR/RGB et statistiques
│   ├── TP1.ipynb                                  # TP1 : Analyse de mesures géométriques et morphologiques
│   ├── Calculs_Mesures_Image_Master.pdf           # Support théorique : Mesures et calculs d'images
│   ├── cours_op_images.pdf                         # Support théorique : Opérations fondamentales sur les images
│   ├── mesures_objets.csv                         # Métriques géométriques d'objets extraites (aire, périmètre, etc.)
│   └── images.jpeg                                # Image de test pour la séance 1
│
├── seance 2/
│   ├── Amine_BOURAMTANE.ipynb                     # Notebook principal : Canaux RGB, Étirement, Égalisation et CLAHE
│   ├── Histogramme.pdf                            # Support théorique : Analyse des histogrammes
│   ├── tp1.pdf                                    # Sujet de travaux pratiques
│   ├── images.jpeg                                # Image d'évaluation (niveaux de gris)
│   ├── imagesCLR.jpeg                             # Image d'évaluation (couleurs)
│   └── images_2.jpeg                              # Image de test additionnelle
│
└── seance 3/
    ├── Tp.ipynb                                   # Notebook principal : Seuillage binaire et algorithme d'Otsu
    ├── TP_Histogramme_Seuillage.pdf               # Sujet de TP : Histogramme et Seuillage
    ├── Seuillage.pdf                              # Support théorique (PDF) : Seuillage d'images
    ├── Seuillage.pptx                             # Support de présentation (PPTX) : Seuillage d'images
    ├── image.png                                  # Image de test pour seuillage
    └── sombre.jpg                                 # Image de test à faible luminosité
```

---

## 4. Contenu des Séances

### Séance 1 : Prise en Main et Mesures Géométriques
- **Objectifs :** Prise en main de la bibliothèque OpenCV, exploration des structures matricielles d'images, manipulation des canaux de couleur et caractérisation géométrique d'objets.
- **Notebooks :**
  - `TP0_Prise_en_main_Vision_Ordinateur.ipynb` : 
    - Vérification de l'environnement de développement.
    - Chargement et conversion d'espaces colorimétriques (`BGR` vs `RGB`, `BGR2GRAY`).
    - Manipulation directe des pixels et séparation des canaux $R$, $G$, $B$.
    - Calcul de statistiques de base (moyenne, écart-type, min/max).
    - Application de filtres moyenneurs ($3 \times 3$ et $5 \times 5$).
  - `TP1.ipynb` :
    - Traitement des mesures géométriques issues d'analyses d'images (`mesures_objets.csv`).
    - Analyse de la circularité et du ratio d'aspect des formes.

### Séance 2 : Opérations Élémentaires et Amélioration d'Histogrammes
- **Objectifs :** Implémentation des opérations arithmétiques ponctuelles, étude et modification dynamique d'histogrammes pour l'amélioration du contraste.
- **Notebook :** `Amine_BOURAMTANE.ipynb`
  - **Exercice 1 :** Séparation des canaux, inversion des niveaux de gris ($255 - I$), modification de luminosité et addition avec saturation (`np.clip`).
  - **Exercice 2 :** Calcul et affichage de l'histogramme de niveaux de gris via `cv2.calcHist` et `plt.hist`.
  - **Exercice 3 :** Étirement linéaire d'histogramme (recadrage de la dynamique sur $[0, 255]$).
  - **Exercice 4 :** Égalisation globale d'histogramme via `cv2.equalizeHist`.
  - **Exercice 5 :** Égalisation adaptative avec limitation de contraste (CLAHE - *Contrast Limited Adaptive Histogram Equalization*).
  - **Exercice 6 :** Traitement des images couleur : comparaison entre égalisation par canal RGB et égalisation du canal de luminance $L$ dans l'espace `CIELAB`.
  - **Exercice 7 :** Discussion comparative et synthèse des performances des méthodes selon les conditions d'éclairage.

### Séance 3 : Histogrammes Avancés et Seuillage
- **Objectifs :** Techniques de segmentation d'image par binarisation manuelle et seuillage automatique.
- **Notebook :** `Tp.ipynb`
  - **Exercice 1 :** Analyse comparée de l'étirement linéaire et de l'égalisation d'histogramme sur des images complexes.
  - **Exercice 2 :** 
    - Seuillage global à valeur fixe ($T = 127, 80, 180$).
    - Seuillage automatique par la méthode d'Otsu (`cv2.THRESH_OTSU`), minimisant la variance intra-classe (ou maximisant la variance inter-classe).

---

## 5. Concepts Théoriques et Algorithmes Implémentés

### 5.1 Transformée d'Histogramme
L'histogramme d'une image numérique représente la distribution des fréquences des niveaux de gris. Pour une image d'intensité $I \in [0, L-1]$ (avec $L = 256$) :

- **Étirement linéaire (Normalisation Min-Max) :**
  $$I_{\text{out}}(x,y) = \frac{I_{\text{in}}(x,y) - I_{\min}}{I_{\max} - I_{\min}} \times 255$$

- **Égalisation d'histogramme (Transformation Cumulée) :**
  Utilise la fonction de distribution cumulée (CDF) $H(i)$ pour transformer les intensités de sorte que l'histogramme résultant soit aussi uniforme que possible :
  $$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j) = \frac{L-1}{N} \sum_{j=0}^{k} n_j$$

- **CLAHE (Contrast Limited Adaptive Histogram Equalization) :**
  Découpe l'image en blocs locaux (par exemple $8 \times 8$) et applique une égalisation d'histogramme avec un seuil de découpage (*clip limit*) pour éviter l'amplification du bruit dans les zones homogènes.

### 5.2 Espace Colorimétrique CIELAB et Traitement Couleur
Pour éviter les distorsions de teinte lors de l'amélioration d'images couleur :
1. Conversion de l'image de `BGR` vers `CIELAB`.
2. Séparation des canaux : $L$ (Luminance), $a$ (Axe rouge-vert), $b$ (Axe jaune-bleu).
3. Application de l'égalisation (ou CLAHE) uniquement sur le canal $L$.
4. Recombinaison des canaux et conversion de retour vers `BGR` / `RGB`.

### 5.3 Seuillage et Algorithme d'Otsu
La binarisation sépare les objets du fond par un seuil $T$ :

$$g(x,y) = \begin{cases} 255 & \text{si } f(x,y) \ge T \\ 0 & \text{si } f(x,y) < T \end{cases}$$

L'**algorithme d'Otsu** sélectionne automatiquement le seuil optimal $T^*$ en maximisant la variance inter-classe $\sigma_B^2(T)$ :

$$\sigma_B^2(T) = \omega_0(T) \cdot \omega_1(T) \cdot \left[ \mu_0(T) - \mu_1(T) \right]^2$$

où $\omega_0, \omega_1$ sont les probabilités des deux classes (fond et forme) et $\mu_0, \mu_1$ leurs moyennes respectives.

### 5.4 Mesures et Descripteurs Géométriques
- **Aire ($A$) :** Nombre total de pixels appartenant à la région de l'objet.
- **Périmètre ($P$) :** Longueur du contour de l'objet.
- **Centroïde ($(cx, cy)$) :** Centre de masse calculé à partir des moments d'image $m_{pq}$ :
  $$cx = \frac{m_{10}}{m_{00}}, \quad cy = \frac{m_{01}}{m_{00}}$$
- **Circularité ($C$) :** Mesure de la régularité de la forme (valeur de 1 pour un cercle parfait) :
  $$C = \frac{4 \pi A}{P^2}$$

---

## 6. Instructions d'Exécution

Pour lancer et exécuter les notebooks Jupyter :

```bash
# Lancement de Jupyter Notebook depuis la racine du projet
jupyter notebook
```

Naviguez ensuite dans le dossier désiré (`seance 1`, `seance 2`, ou `seance 3`) et ouvrez le fichier `.ipynb` correspondant. Exécutez les cellules séquentiellement (`Shift + Enter`).

---

## 7. Auteur et Contact

- **Auteur :** Amine BOURAMTANE
- **Dépôt GitHub :** [vision-par-ordinateur](https://github.com/Aminebouramtane/vision-par-ordinateur)
