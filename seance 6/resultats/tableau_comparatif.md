| Méthode | Continuité des contours | Sensibilité aux textures/bruit | Coût moyen CPU | Observations |
|:--|:--|:--|--:|:--|
| Sobel | Bandes larges ; pas de chaînage explicite | Fortes réponses aux variations locales ; bruit atténué par le flou | 2.919 ms | Visualise la force du gradient, sans binarisation |
| Canny (100, 200) | Lignes fines ; ruptures possibles aux faibles contrastes | Détails contrastés conservés ; seuils et flou réduisent les réponses faibles | 3.663 ms | Rapide ; compromis précision spatiale / perte de détails |
| HED | Grandes structures plus larges et progressives | Atténuation de certains détails ; dépend du domaine appris | 1822.605 ms | Carte continue ; contours parfois épais ; pas d'immunité au bruit |
| Fusion α=0,6 | Superposition de lignes fines et bandes HED | Peut réintroduire les détails et défauts des deux cartes | 1791.275 ms | Compromis visuel ; coût dominé par HED |