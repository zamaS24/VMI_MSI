# Explicabilite des modeles de classification de textes - Approche BERT

Ce dossier contient le pipeline BERT/CamemBERT du projet de classification binaire de textes `homme` vs `femme`.

L'objectif n'est pas seulement de predire une classe, mais aussi de comprendre les indices utilises par le modele pour produire ses decisions. Le projet combine donc entrainement, evaluation, explicabilite et visualisations, afin de servir a la fois de code source reproductible et de mini-rapport de projet.

> Les resultats presentes ici decrivent le comportement du modele sur ce dataset. Ils ne doivent pas etre interpretes comme des regles generales sur l'ecriture masculine ou feminine.

## Sommaire

- [1. Sujet du projet](#1-sujet-du-projet)
- [2. Organisation du dossier](#2-organisation-du-dossier)
- [3. Dataset](#3-dataset)
- [4. Pipeline global](#4-pipeline-global)
- [5. Modele et entrainement](#5-modele-et-entrainement)
- [6. Resultats](#6-resultats)
- [7. Explicabilite](#7-explicabilite)
- [8. Visualisation locale et attention](#8-visualisation-locale-et-attention)
- [9. Scripts et commandes](#9-scripts-et-commandes)
- [10. Limites](#10-limites)

## 1. Sujet du projet

Le projet etudie une tache de classification binaire de textes :

| Element | Description |
| --- | --- |
| Entree | Fichiers texte bruts |
| Sortie | Classe predite : `femme` ou `homme` |
| Modele | CamemBERT pour classification de sequences |
| Evaluation | Accuracy, precision, recall, F1-score, matrice de confusion, ROC |
| Explicabilite | SHAP, LIME, explications locales, poids d'attention |

Le modele utilise une representation contextuelle BERT, contrairement a une approche TF-IDF ou chaque feature correspond directement a un mot. Ici, les representations dependent du contexte, ce qui rend le modele plus puissant mais aussi plus difficile a interpreter.

## 2. Organisation du dossier

```text
BERT/
  config.py                 # chemins, labels, hyperparametres par defaut
  data_loader.py            # chargement dataset, extraction labels, chunking CamemBERT
  model.py                  # creation, sauvegarde, chargement et prediction du modele
  train.py                  # fine-tuning complet de CamemBERT
  evaluate.py               # evaluation sur le split test
  explain_shap.py           # explications SHAP
  explain_lime.py           # explications LIME
  experiment.py             # runner experimental pour comparer methodes/modeles
  utils.py                  # metriques, plots, JSON, seeds
  everything.ipynb          # notebook d'exploration et consolidation des experiences
  requirements.txt          # dependances Python
  outputs/                  # modeles sauvegardes, checkpoints, logs
  artifacts/                # resultats tabulaires
  vis/                      # visualisations generees par les scripts
  presentation/             # presentation Beamer et figures du rapport
```

Les figures utilisees dans ce README proviennent principalement de :

```text
BERT/presentation/images/
```

## 3. Dataset

Le dataset est organise en trois splits :

| Split | Total | Femme | Homme |
| --- | ---: | ---: | ---: |
| Train | 852 | 492 | 360 |
| Validation | 284 | 164 | 120 |
| Test | 285 | 165 | 120 |

Les labels sont extraits automatiquement depuis les noms des fichiers. Dans le code, le quatrieme champ parenthese du nom de fichier est utilise :

- valeur `1` : classe `homme`
- valeur `2` : classe `femme`

Si cette convention n'est pas disponible, le code essaie aussi d'inferer le label depuis les dossiers parents `homme` ou `femme`.

<p align="center">
  <img src="presentation/images/distribution_total.png" alt="Distribution totale homme femme" width="44%">
  <img src="presentation/images/distribution_splits.png" alt="Distribution par split" width="44%">
</p>

### Chunking des documents longs

CamemBERT accepte des sequences de longueur limitee. Les textes longs sont donc decoupes en chunks de longueur maximale 512 tokens, tokens speciaux inclus.

Dans l'experience presentee, l'echantillonnage des chunks est :

| Classe | Nombre de chunks par document |
| --- | ---: |
| Homme | 64 |
| Femme | 32 |

Ce choix est utilise car il y a davantage de documents `femme` que de documents `homme`. Le nombre de chunks par document est donc ajuste pour reduire le desequilibre pendant l'entrainement.

Important : pour l'evaluation document-level, les probabilites sont calculees sur les chunks puis moyennees afin d'obtenir une prediction finale par document.

## 4. Pipeline global

Le pipeline suit les etapes suivantes :

```text
Textes bruts
  -> extraction des labels
  -> tokenisation CamemBERT
  -> decoupage en chunks
  -> classification avec CamemBERT
  -> evaluation
  -> explicabilite SHAP/LIME
  -> explications locales et attention weights
```

![Pipeline global BERT](presentation/images/overview.png)

## 5. Modele et entrainement

### Architecture

Le modele repose sur `camembert-base` avec une tete de classification binaire. Les representations contextuelles produites par CamemBERT sont envoyees vers une couche de classification qui produit deux logits : `femme` et `homme`.

![Architecture du reseau](presentation/images/architecture_reseau.png)

### Regimes compares

Deux regimes experimentaux sont presentes :

| Regime | Principe |
| --- | --- |
| Fine-tuning complet | Tous les parametres du modele BERT sont adaptes au dataset |
| Linear probing | Le backbone sert d'extracteur de representations, et la classification est surtout portee par la tete lineaire |

Le fine-tuning complet est plus flexible, mais plus couteux et plus sensible au surapprentissage. Le linear probing est plus contraint, mais permet d'observer si les representations pre-entrainees contiennent deja assez d'information pour la tache.

### Hyperparametres principaux

| Hyperparametre | Valeur |
| --- | ---: |
| Batch size | 32 |
| Batch size evaluation | 32 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Max length | 512 tokens |
| Seed | 42 |

Le script d'entrainement utilise aussi un scheduler lineaire avec warmup, du gradient clipping et la sauvegarde du meilleur modele.

## 6. Resultats

### Fine-tuning complet

Le premier entrainement correspond a un fine-tuning complet de CamemBERT.

<p align="center">
  <img src="presentation/images/finetuning_loss.png" alt="Courbe de loss fine-tuning" width="48%">
  <img src="presentation/images/matrice_confusion_finetuning.png" alt="Matrice de confusion fine-tuning" width="48%">
</p>

### Linear probing

Le second entrainement correspond au linear probing.

<p align="center">
  <img src="presentation/images/linear_probing_loss.png" alt="Courbe de loss linear probing" width="46%">
  <img src="presentation/images/confusion_matrix_linear_probing.png" alt="Matrice de confusion linear probing" width="46%">
</p>

Les resultats presentes pour le linear probing sont :

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| femme | 82.1% | 80.6% | 81.3% |
| homme | 74.0% | 75.8% | 74.9% |
| Accuracy globale |  |  | 78.6% |

La courbe ROC complete l'evaluation en montrant le compromis entre vrais positifs et faux positifs.

<p align="center">
  <img src="presentation/images/roc_curve.png" alt="Courbe ROC" width="55%">
</p>

## 7. Explicabilite

L'explicabilite est centree sur deux methodes :

| Methode | Idee principale |
| --- | --- |
| SHAP | Estimer l'importance des tokens par contribution marginale au score de prediction |
| LIME | Approximer localement le modele par un modele interpretable autour d'un exemple |

Ces deux methodes sont complementaires : si elles font ressortir des indices lexicaux proches, l'explication devient plus credible que si une seule methode est utilisee.

<p align="center">
  <img src="presentation/images/SHAP.png" alt="Principe SHAP" width="44%">
  <img src="presentation/images/LIME.png" alt="Principe LIME" width="44%">
</p>

### Explications globales SHAP

Les figures suivantes presentent les termes qui soutiennent les predictions du modele pour chaque classe selon SHAP.

<p align="center">
  <img src="presentation/images/shap_top_homme_terms.png" alt="SHAP top termes homme" width="46%">
  <img src="presentation/images/shap_top_femme_terms.png" alt="SHAP top termes femme" width="46%">
</p>

### Explications globales LIME

Les figures suivantes presentent les termes qui soutiennent les predictions du modele pour chaque classe selon LIME.

<p align="center">
  <img src="presentation/images/lime_top_homme_terms.png" alt="LIME top termes homme" width="46%">
  <img src="presentation/images/lime_top_femme_terms.png" alt="LIME top termes femme" width="46%">
</p>

### Comparaison SHAP/LIME

Les termes communs entre SHAP et LIME indiquent les indices lexicaux les plus stables pour chaque classe.

| Classe | Termes communs |
| --- | --- |
| Homme | `je`, `de`, `il`, `j` |
| Femme | `l`, `elle`, `les`, `marie`, `suis`, `d`, `seule` |

On observe une convergence partielle entre les deux methodes. Ces termes restent toutefois lies au dataset, au modele et au protocole d'entrainement.

## 8. Visualisation locale et attention

Les explications globales donnent une vision moyenne du comportement du modele. Les explications locales montrent, pour un texte precis, les tokens qui ont pousse la prediction dans un sens ou dans l'autre.

<p align="center">
  <img src="presentation/images/shap_local_explanation.png" alt="Explication locale SHAP" width="46%">
  <img src="presentation/images/lime_local_explanation.png" alt="Explication locale LIME" width="46%">
</p>

Une autre visualisation utile consiste a afficher directement les scores d'attribution sur le texte.

<p align="center">
  <img src="presentation/images/text_highlight.png" alt="Scores d'attribution sur texte" width="70%">
</p>

Les poids d'attention peuvent aussi etre inspectes. Ils ne constituent pas une preuve causale d'explication, mais ils donnent une indication supplementaire sur les relations token-token auxquelles le modele accorde du poids.

<p align="center">
  <img src="presentation/images/attention_weights.png" alt="Poids d'attention" width="70%">
</p>

## 9. Scripts et commandes

Les commandes ci-dessous se lancent depuis la racine du depot.

### Installation

```bash
pip install -r BERT/requirements.txt
```

Si l'environnement conda du projet est utilise :

```bash
conda activate train
pip install -r BERT/requirements.txt
```

### Entrainement

```bash
python BERT/train.py --data-dir data/datasetSujet3/content/dataset --batch-size 32 --eval-batch-size 32 --epochs 5 --learning-rate 2e-5 --num-chunks-homme 64 --num-chunks-femme 32
```

Sorties principales :

```text
BERT/outputs/models/best_model/
BERT/outputs/checkpoints/
BERT/outputs/logs/history.csv
BERT/artifacts/metrics.json
```

### Evaluation

```bash
python BERT/evaluate.py --data-dir data/datasetSujet3/content/dataset --model-dir BERT/outputs/models/best_model --num-chunks-homme 64 --num-chunks-femme 32
```

Sorties attendues :

```text
BERT/artifacts/test_predictions.csv
BERT/artifacts/metrics.json
BERT/vis/confusion_matrix.png
BERT/vis/roc_curve.png
```

### Explications SHAP

```bash
python BERT/explain_shap.py --data-dir data/datasetSujet3/content/dataset --model-dir BERT/outputs/models/best_model --split test --n-examples 20 --n-terms 20 --num-chunks-homme 64 --num-chunks-femme 32
```

### Explications LIME

```bash
python BERT/explain_lime.py --data-dir data/datasetSujet3/content/dataset --model-dir BERT/outputs/models/best_model --split test --n-examples 50 --n-terms 20 --num-samples 500 --num-chunks-homme 64 --num-chunks-femme 32
```

### Role des fichiers sources

| Fichier | Role |
| --- | --- |
| `config.py` | centralise les chemins, labels et hyperparametres |
| `data_loader.py` | charge les textes, extrait les labels, construit les chunks |
| `model.py` | cree CamemBERT, charge le modele sauvegarde, moyenne les predictions par chunks |
| `train.py` | entraine le modele et sauvegarde le meilleur checkpoint |
| `evaluate.py` | calcule les metriques et les figures d'evaluation |
| `explain_shap.py` | genere les attributions SHAP locales et globales |
| `explain_lime.py` | genere les attributions LIME locales et globales |
| `experiment.py` | permet de comparer plusieurs configurations experimentales |
| `utils.py` | fonctions communes : metriques, plots, seeds, JSON |
| `everything.ipynb` | notebook d'exploration et de consolidation des experiences |

## 10. Limites

- Les labels sont extraits depuis les noms de fichiers : si cette convention contient un biais, le modele peut l'apprendre indirectement.
- Les textes longs sont decoupes en chunks : le choix du nombre de chunks par classe influence les donnees vues par le modele.
- SHAP et LIME fournissent des approximations du comportement du modele, pas des preuves causales.
- Les poids d'attention sont utiles pour inspecter le modele, mais ne suffisent pas a eux seuls a expliquer une decision.
- Les termes mis en avant sont specifiques a ce dataset et a ce modele.

## Conclusion

L'approche BERT/CamemBERT fournit un classifieur plus contextuel qu'une representation TF-IDF classique. Les resultats montrent que le modele apprend des signaux utiles pour distinguer les deux classes, et les methodes SHAP/LIME permettent d'inspecter les tokens qui influencent ses decisions.

Le projet montre donc une chaine complete :

```text
classification -> evaluation -> explicabilite globale -> explicabilite locale -> interpretation critique
```

Cette lecture reste volontairement limitee au comportement du classifieur sur ce dataset.
