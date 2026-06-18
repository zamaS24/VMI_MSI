# TF-IDF et explicabilite des modeles

Ce dossier contient le pipeline TF-IDF du projet de classification binaire de textes `homme` vs `femme`.

L'objectif est double :

1. entrainer un classifieur texte simple et interpretable ;
2. verifier si les termes identifies par les methodes d'explicabilite influencent reellement le comportement du modele.

Le projet combine donc code source, evaluation, visualisations et validation comportementale des explications.

> Les resultats presentes ici decrivent le comportement du classifieur sur ce dataset. Ils ne doivent pas etre interpretes comme des regles generales sur l'ecriture masculine ou feminine.

## Sommaire

- [1. Sujet du projet](#1-sujet-du-projet)
- [2. Organisation du dossier](#2-organisation-du-dossier)
- [3. Dataset](#3-dataset)
- [4. Pipeline global](#4-pipeline-global)
- [5. Representation TF-IDF](#5-representation-tf-idf)
- [6. Modele MLP](#6-modele-mlp)
- [7. Resultats de classification](#7-resultats-de-classification)
- [8. Explicabilite](#8-explicabilite)
- [9. Perturbations](#9-perturbations)
- [10. Scripts et commandes](#10-scripts-et-commandes)
- [11. Limites](#11-limites)

## 1. Sujet du projet

La tache consiste a classifier des textes en deux classes :

| Element | Description |
| --- | --- |
| Entree | Fichiers texte bruts |
| Sortie | Classe predite : `femme` ou `homme` |
| Representation | TF-IDF unigrammes |
| Classifieur | MLP PyTorch |
| Evaluation | Accuracy, precision, recall, F1-score, matrice de confusion |
| Explicabilite | Integrated Gradients, LRP |
| Validation | Perturbations `remove`, `mask`, `swap` |

L'interet de TF-IDF est que chaque dimension correspond directement a un terme du vocabulaire. Cela rend le lien entre une feature et un mot beaucoup plus direct que dans une representation contextuelle de type BERT.

## 2. Organisation du dossier

```text
TFIDF/
  data_loader.py                    # chargement des splits et extraction des labels
  utils.py                          # TF-IDF, labels, sauvegarde JSON/pickle, seeds
  model.py                          # architecture MLP courante
  train.py                          # entrainement, evaluation, courbes de loss
  explain_integrated_gradients.py   # explications IG locales et globales
  explain_lrp.py                    # explications LRP locales et globales
  perturbation_test.py              # tests remove/mask/swap sur termes explicatifs
  visualize_text_explanations.py    # visualisation HTML des scores sur le texte
  main.ipynb                        # notebook source de reference du travail
  artifacts/                        # metriques et predictions
  outputs/                          # modele et vectorizer sauvegardes
  vis/                              # figures et resultats generes par les scripts
  vis_old/                          # anciens graphes et sorties historiques
  presentations/                    # presentation Beamer et images illustrees
```

Les figures du rapport GitHub utilisent principalement :

```text
TFIDF/presentations/images/
TFIDF/vis/
```

## 3. Dataset

Le dataset est divise en trois splits :

| Split | Total | Femme | Homme |
| --- | ---: | ---: | ---: |
| Train | 852 | 492 | 360 |
| Validation | 284 | 164 | 120 |
| Test | 285 | 165 | 120 |

Les labels sont extraits depuis les noms de fichiers. Le code lit le quatrieme champ entre parentheses :

- `1` correspond a `homme` ;
- `2` correspond a `femme`.

<p align="center">
  <img src="presentations/images/distribution_total.png" alt="Distribution totale homme femme" width="44%">
  <img src="presentations/images/distribution_splits.png" alt="Distribution des classes par split" width="44%">
</p>

Ce point est important : si les fichiers ou la constitution du corpus contiennent des biais, le modele peut les apprendre. L'analyse doit donc porter sur le comportement du modele dans ce contexte experimental, pas sur des proprietes universelles du langage.

## 4. Pipeline global

Le pipeline TF-IDF suit cette logique :

```text
Textes bruts
  -> extraction des labels
  -> vectorisation TF-IDF
  -> MLP PyTorch
  -> evaluation
  -> explications IG + LRP
  -> perturbations remove/mask/swap
  -> visualisation des attributions sur le texte
```

![Pipeline TF-IDF](presentations/images/overview.png)

## 5. Representation TF-IDF

Le fichier `utils.py` definit les parametres TF-IDF par defaut :

| Parametre | Valeur courante |
| --- | --- |
| `max_features` | 25000 |
| `min_df` | 2 |
| `max_df` | 0.85 |
| `ngram_range` | `(1, 1)` |

Les resultats sauvegardes dans `artifacts/metrics.json` correspondent a une execution avec :

| Element | Valeur sauvegardee |
| --- | ---: |
| `max_features` | 21000 |
| `min_df` | 2 |
| `max_df` | 0.85 |
| Nombre de features retenues | 21000 |

Le choix de limiter le vocabulaire est important. Sans limite, le nombre de features peut devenir tres grand et favoriser le surapprentissage, surtout avec un dataset de taille moderee.

## 6. Modele MLP

Le modele courant dans `model.py` est un petit MLP :

```text
Vecteur TF-IDF
  -> Linear
  -> BatchNorm1d
  -> ReLU
  -> Dropout
  -> Linear
  -> BatchNorm1d
  -> ReLU
  -> Dropout
  -> Linear
  -> logits
```

Architecture courante du code :

| Element | Valeur |
| --- | --- |
| Couches cachees | `(64, 32)` |
| Normalisation | `BatchNorm1d` |
| Activation | `ReLU` |
| Dropout | `0.2` |
| Sortie | 2 logits |

Deux architectures ont ete comparees pendant le developpement. La premiere montrait un surapprentissage net. La seconde reduit la capacite et ajoute davantage de regularisation.

<p align="center">
  <img src="presentations/images/architecture_1.png" alt="Architecture initiale" width="46%">
  <img src="presentations/images/architecture_2_clean.png" alt="Architecture reduisant overfitting" width="46%">
</p>

Les courbes ci-dessous illustrent cette phase de diagnostic : la loss validation permet de detecter le surapprentissage, puis de garder le meilleur checkpoint validation.

<p align="center">
  <img src="vis/tfidf_loss_curve.png" alt="Courbe de loss TF-IDF" width="46%">
  <img src="presentations/images/nouvelle_courbe_loss.png" alt="Nouvelle courbe de loss" width="46%">
</p>

## 7. Resultats de classification

Les resultats sauvegardes dans `artifacts/metrics.json` donnent :

| Metrique | Valeur |
| --- | ---: |
| Accuracy test | 0.853 |
| Macro F1 | 0.850 |
| Weighted F1 | 0.853 |
| Test loss | 0.344 |

Detail par classe :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| femme | 0.882 | 0.861 | 0.871 | 165 |
| homme | 0.815 | 0.842 | 0.828 | 120 |

Matrice de confusion sauvegardee :

```text
               Pred femme   Pred homme
True femme          142          23
True homme           19         101
```

<p align="center">
  <img src="presentations/images/matrice_confusion.png" alt="Matrice de confusion TF-IDF" width="55%">
</p>

Ces performances indiquent que la baseline TF-IDF + MLP est solide, mais le coeur du projet est l'explication du comportement du modele.

## 8. Explicabilite

Deux methodes sont utilisees :

| Methode | Principe |
| --- | --- |
| Integrated Gradients | Mesure la contribution de chaque feature TF-IDF en allant d'une baseline zero vers l'entree reelle |
| LRP | Redistribue le score de prediction depuis la sortie du reseau vers les features d'entree |

<p align="center">
  <img src="presentations/images/integrated_gradients.png" alt="Integrated Gradients" width="44%">
  <img src="presentations/images/LRP.png" alt="LRP" width="44%">
</p>

### Explications globales

Les explications globales agregent les attributions sur les textes correctement classes. Les figures suivantes montrent les termes qui soutiennent les predictions du modele, pas des termes qui definiraient les classes dans l'absolu.

<p align="center">
  <img src="vis/tfidf_ig_top_homme_terms.png" alt="IG top termes homme" width="46%">
  <img src="vis/tfidf_ig_top_femme_terms.png" alt="IG top termes femme" width="46%">
</p>

<p align="center">
  <img src="vis/tfidf_lrp_top_homme_terms.png" alt="LRP top termes homme" width="46%">
  <img src="vis/tfidf_lrp_top_femme_terms.png" alt="LRP top termes femme" width="46%">
</p>

### Convergence IG/LRP

Comparer IG et LRP permet de chercher des signaux plus stables. Dans la presentation, les termes communs retenus sont :

| Classe | Termes communs IG/LRP |
| --- | --- |
| Homme | `fit`, `jules` |
| Femme | `marie`, `fanny`, `tante` |

L'intersection entre IG et LRP permet d'identifier les termes les plus stables pour chaque classe. Les termes communs montrent que les deux methodes convergent partiellement vers les memes indices lexicaux.

### Explications locales

Les fichiers locaux produits par IG et LRP permettent d'inspecter un texte particulier : vrai label, prediction, confiance, top termes positifs et top termes negatifs.

Une visualisation HTML permet ensuite de colorer directement le texte selon les scores d'attribution.

<p align="center">
  <img src="presentations/images/example_viz_score.png" alt="Visualisation des scores d'attribution" width="70%">
</p>

## 9. Perturbations

Les perturbations servent a verifier si les termes explicatifs affectent reellement le comportement du modele.

Protocole :

1. selectionner des textes correctement classes ;
2. recuperer les termes explicatifs IG, LRP ou leur intersection ;
3. perturber ces termes dans le texte ;
4. relancer le modele ;
5. mesurer la baisse d'accuracy, la baisse de confiance et le nombre de predictions inversees.

Modes de perturbation :

| Mode | Description |
| --- | --- |
| `remove` | supprimer les termes explicatifs |
| `mask` | remplacer les termes par `[MASK]` |
| `swap` | remplacer des termes associes a une classe par des termes associes a l'autre classe |

### Resultats des 9 combinaisons

Les resultats ci-dessous correspondent a 100 textes correctement classes, 50 par classe, avec 20 termes explicatifs par classe.

| Methode | Mode | Accuracy avant | Accuracy apres | Drop accuracy | Drop confiance | Predictions inversees |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IG | mask | 1.00 | 0.97 | 0.03 | 0.022 | 3 |
| IG | remove | 1.00 | 0.97 | 0.03 | 0.022 | 3 |
| IG | swap | 1.00 | 0.90 | 0.10 | 0.076 | 10 |
| Intersection | mask | 1.00 | 0.99 | 0.01 | 0.003 | 1 |
| Intersection | remove | 1.00 | 0.99 | 0.01 | 0.003 | 1 |
| Intersection | swap | 1.00 | 0.98 | 0.02 | 0.009 | 2 |
| LRP | mask | 1.00 | 1.00 | 0.00 | 0.001 | 0 |
| LRP | remove | 1.00 | 1.00 | 0.00 | 0.001 | 0 |
| LRP | swap | 1.00 | 0.97 | 0.03 | 0.009 | 3 |

Lecture : si la suppression, le masquage ou le remplacement des termes explicatifs diminue l'accuracy ou la confiance, alors ces termes ont une importance comportementale pour ce modele.

Exemples de figures de perturbation :

<p align="center">
  <img src="vis/tfidf_perturbation_accuracy_drop_method-ig_mode-swap_docs-100_per-class-50_terms-20.png" alt="Accuracy drop IG swap" width="46%">
  <img src="vis/tfidf_perturbation_confidence_drop_method-ig_mode-swap_docs-100_per-class-50_terms-20.png" alt="Confidence drop IG swap" width="46%">
</p>

## 10. Scripts et commandes

Les commandes ci-dessous se lancent depuis la racine du depot.

### Installation minimale

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn joblib
```

Avec l'environnement conda du projet :

```bash
conda activate train
```

### Entrainement

```bash
python TFIDF/train.py --base-dir data/datasetSujet3/content/dataset --output-dir TFIDF/outputs --artifact-dir TFIDF/artifacts --confusion-matrix-path TFIDF/vis/tfidf_confusion_matrix.png --loss-plot-path TFIDF/vis/tfidf_loss_curve.png --max-features 25000 --min-df 2 --max-df 0.85 --no-show-plot
```

Sorties principales :

```text
TFIDF/outputs/tf_idf_mlp_model.pt
TFIDF/outputs/tfidf_vectorizer.pkl
TFIDF/outputs/history.csv
TFIDF/outputs/loss_curve.png
TFIDF/artifacts/metrics.json
TFIDF/artifacts/test_predictions.csv
TFIDF/vis/tfidf_confusion_matrix.png
```

### Integrated Gradients

```bash
python TFIDF/explain_integrated_gradients.py --data-dir data/datasetSujet3/content/dataset --model-path TFIDF/outputs/tf_idf_mlp_model.pt --vectorizer-path TFIDF/outputs/tfidf_vectorizer.pkl --global-output TFIDF/vis/tfidf_integrated_gradients_global.csv --local-output TFIDF/vis/tfidf_integrated_gradients_local.csv --homme-plot TFIDF/vis/tfidf_ig_top_homme_terms.png --femme-plot TFIDF/vis/tfidf_ig_top_femme_terms.png --top-k 20
```

### LRP

```bash
python TFIDF/explain_lrp.py --data-dir data/datasetSujet3/content/dataset --model-path TFIDF/outputs/tf_idf_mlp_model.pt --vectorizer-path TFIDF/outputs/tfidf_vectorizer.pkl --global-output TFIDF/vis/tfidf_lrp_global.csv --local-output TFIDF/vis/tfidf_lrp_local.csv --homme-plot TFIDF/vis/tfidf_lrp_top_homme_terms.png --femme-plot TFIDF/vis/tfidf_lrp_top_femme_terms.png --top-k 20
```

### Perturbation d'une combinaison

```bash
python TFIDF/perturbation_test.py --method ig --mode swap --data-dir data/datasetSujet3/content/dataset --model-path TFIDF/outputs/tf_idf_mlp_model.pt --vectorizer-path TFIDF/outputs/tfidf_vectorizer.pkl --ig-global-path TFIDF/vis/tfidf_integrated_gradients_global.csv --lrp-global-path TFIDF/vis/tfidf_lrp_global.csv --output-dir TFIDF/vis --n-texts-per-class 50 --n-terms 20
```

### Toutes les combinaisons de perturbation

PowerShell :

```powershell
foreach ($method in @("ig", "lrp", "intersection")) {
  foreach ($mode in @("remove", "mask", "swap")) {
    python TFIDF/perturbation_test.py --method $method --mode $mode --data-dir data/datasetSujet3/content/dataset --model-path TFIDF/outputs/tf_idf_mlp_model.pt --vectorizer-path TFIDF/outputs/tfidf_vectorizer.pkl --ig-global-path TFIDF/vis/tfidf_integrated_gradients_global.csv --lrp-global-path TFIDF/vis/tfidf_lrp_global.csv --output-dir TFIDF/vis --n-texts-per-class 50 --n-terms 20
  }
}
```

### Visualisation HTML des scores

```bash
python TFIDF/visualize_text_explanations.py --method both --scope local --row-index 0 --n-terms 20 --output-dir TFIDF/vis/text_highlights
```

### Role des fichiers sources

| Fichier | Role |
| --- | --- |
| `data_loader.py` | charge les splits, lit les textes et extrait les labels |
| `utils.py` | construit les features TF-IDF, encode les labels, sauvegarde JSON/pickle |
| `model.py` | definit le MLP PyTorch |
| `train.py` | entraine le modele, evalue le test set, sauvegarde courbes et metriques |
| `explain_integrated_gradients.py` | genere les explications IG locales/globales |
| `explain_lrp.py` | genere les explications LRP locales/globales |
| `perturbation_test.py` | applique remove/mask/swap et mesure l'impact |
| `visualize_text_explanations.py` | produit les visualisations HTML sur texte |
| `main.ipynb` | notebook source de reference et d'experimentation |

## 11. Limites

- TF-IDF ignore l'ordre des mots et la syntaxe.
- Le vocabulaire TF-IDF depend fortement du corpus et des parametres `min_df`, `max_df`, `max_features`.
- Les explications IG et LRP sont specifiques a ce modele et a cette vectorisation.
- LRP a travers BatchNorm et Dropout est traite de maniere approximative dans le script.
- Les perturbations peuvent produire des textes peu naturels.
- Les termes mis en avant ne doivent pas etre interpretes comme des marqueurs universels du genre.

## Conclusion

Le pipeline TF-IDF + MLP fournit une baseline interpretable et performante pour cette tache. Les methodes Integrated Gradients et LRP permettent d'identifier des termes qui influencent les predictions, puis les tests de perturbation verifient si ces termes modifient effectivement le comportement du modele.

La contribution principale du projet est donc la chaine complete :

```text
classification -> explication -> perturbation -> validation comportementale
```

Cette lecture reste limitee au classifieur entraine et au dataset utilise.
