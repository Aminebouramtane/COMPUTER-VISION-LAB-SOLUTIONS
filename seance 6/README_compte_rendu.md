# TP : détection de contours

Le fichier principal est `Compte_rendu_detection_contours.ipynb` : il contient le code, les résultats exécutés, les analyses des parties A à D et les réponses aux huit questions. Compléter les noms, le groupe et la date dans la première cellule.

L'image utilisée est une photographie publique NASA (Eileen Collins), distribuée par scikit-image. Les observations se rapportent à cette image. Les sources et les limites de l'expérience sont précisées dans le notebook.

## Lecture

Ouvrir le notebook dans Jupyter ou VS Code. Les sorties sont déjà intégrées. La version HTML permet une lecture sans exécution ; les formules peuvent nécessiter Internet pour le chargement de MathJax.

## Réexécution

Utiliser Python 3.12, puis depuis ce dossier :

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Ouvrir le notebook et lancer toutes les cellules. Pour le script autonome :

```bash
python tp_contours.py
```

Conserver `image.jpg`, `deploy.prototxt` et `hed_pretrained_bsds.caffemodel` dans le même dossier. OpenCV 4.12 est fixé car le chargeur Caffe requis n'est pas disponible dans la version 5 testée. Le notebook fonctionne sur CPU ; aucun GPU ni entraînement n'est nécessaire.

Les fichiers `resultats/` comprennent les cartes PNG, les planches comparatives et les mesures CSV. Le modèle provient d'un miroir identifié dans le notebook ; son empreinte SHA-256 est affichée lors de l'exécution.

Si l'image ou les paramètres sont changés, réexécuter toutes les cellules et adapter les observations rédigées. Les temps dépendent de la machine ; la densité des cartes n'est pas une métrique de justesse.
