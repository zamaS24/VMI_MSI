# Tasks 

## TF-IDF : 
- [ ] les viz sur les texts 




## BERT : 
- [ ] Attention Weights
- 






# Recap 
# TFIDF
    - perturbations are done on the test set ! 
    - we select only the documents that have classified with accuracy 1.0 before running perturbation tests 
    In TF-IDF perturbation, the script uses the test split, predicts all test documents with the trained TF-IDF + MLP model, then selects only documents that
    were already correctly classified.
    - now we changed, we run on chunks

      So:

    - explained_class = homme
        positive score  -> supports homme
        negative score  -> opposes homme
        And because binary classifier:
        opposes homme ~= supports femme
        But I would describe it carefully as “evidence against homme”, not automatically “femme word.”




# Commands : 
running perturbations : 
  $methods = @("ig", "lrp", "intersection")
  $modes = @("remove", "mask", "swap")

  foreach ($method in $methods) {
    foreach ($mode in $modes) {
      python .\TFIDF\perturbation_test.py `
        --method $method `
        --mode $mode `
        --data-dir .\data\datasetSujet3\content\dataset `
        --model-path .\TFIDF\outputs\tf_idf_mlp_model.pt `
        --vectorizer-path .\TFIDF\outputs\tfidf_vectorizer.pkl `
        --ig-global-path .\TFIDF\vis\tfidf_integrated_gradients_global.csv `
        --lrp-global-path .\TFIDF\vis\tfidf_lrp_global.csv `
        --output-dir .\TFIDF\vis `
        --n-texts-per-class 50 `
        --n-terms 20
    }
  }









# NOTES : 
Modifie la présentation comme suit :

1. Slide n°4, juste après la slide “Context”
   - Supprimer la partie “Positionnement”.
   - Ajouter un placeholder pour une image globale.

2. Slide “Dataset”
   - Ajouter l’image de la distribution totale homme/femme.
   - Ajouter l’image de la distribution des splits.
   - Les deux images se trouvent dans le dossier `TFIDF/presentation/images/`.

3. Slide “Pipeline”
   - Supprimer le texte situé après le diagramme.

4. Slide “Représentation TF-IDF”
   - Supprimer entièrement cette slide.

5. Section “Architecture du classifieur”

   Architecture 1 :
   - Ajouter une slide contenant uniquement l’image de l’architecture :
     `TFIDF/presentation/images/architecture_1.png`
   - Ajouter ensuite une slide avec les courbes d’entraînement actuelles.
   - Conserver l’image actuelle des courbes, car elle montre l’overfitting.

   Architecture pour réduire l’overfitting :
   - Ajouter une slide avec la nouvelle architecture :
     `TFIDF/presentation/images/architecture_2.png`
   - Ajouter les nouvelles courbes d’entraînement :
     `TFIDF/presentation/images/nouvelle_courbe_loss.png`

6. Évaluation
   - Ajouter la matrice de confusion :
     `TFIDF/presentation/images/matrice_confusion.png`

7. Section “Explicabilité”
   - Présenter deux méthodes : Integrated Gradients et LRP.
   - Supprimer le texte explicatif existant.
   - Ajouter une slide “Integrated Gradients” contenant uniquement l’image :
     `TFIDF/presentation/images/integrated_gradients.png`
   - Ajouter une slide “LRP” contenant uniquement l’image :
     `TFIDF/presentation/images/LRP.png`
   - Conserver les slides “Explication globale IG” et “Explication globale LRP”.
   - Renommer la slide “Cohésion entre IG et LRP” avec un terme plus approprié, par exemple :
     “Comparaison entre IG et LRP” ou “Convergence entre IG et LRP”.

8. Section “Perturbations”
   - Sauter directement aux perturbations.
   - Après la slide “Protocole de perturbation”, ajouter :
     - un graphique d’accuracy drop ;
     - un graphique de confidence drop.
   - Utiliser les graphes situés dans `TFIDF/vis_old/`.
   - Conserver le tableau juste après ces figures.
   - La figure “accuracy avant vs après” doit apparaître avant le tableau.

9. Visualisation des scores d’attribution
   - Ajouter une slide indiquant que les scores d’attribution sont visualisés directement sur le texte.
   - Utiliser l’image :
     `example_viz_score.png`

10. Conclusion
   - Conserver la conclusion existante.




-
-
-
-

- sur la slide 4 : 
  - enlève tache  et figure a completer 
  - tu met's l'image de 
  - TFIDF/presentatoins/images_presentation/overview.png
  - les objectifs tu les mets avec un font pas très très grand ! 


Pour la slide “Analyse globale / Table à compléter”, complète le contenu à partir des résultats IG et LRP.

Objectif de la slide :
montrer les termes communs entre les deux méthodes d’explicabilité, Integrated Gradients et LRP, afin d’identifier les signaux lexicaux les plus robustes utilisés par le modèle.

Remplacer le titre par :
“Convergence des attributions IG et LRP”

Dans la partie gauche, mettre :

Analyse globale

• Comparer les termes les plus attribués par IG et LRP.
• Identifier les termes communs par classe.
• Utiliser l’intersection comme signal robuste.
• Vérifier si les deux méthodes convergent vers les mêmes indices lexicaux.

Dans le tableau de droite, compléter comme suit :

Classe | Termes communs IG/LRP
Homme  | fit, jules
Femme  | marie, fanny, tante

Ajouter en bas de la slide, si l’espace le permet :

Les termes communs montrent que IG et LRP attribuent de l’importance à des indices lexicaux similaires.
La présence de prénoms et de marqueurs sociaux suggère que le modèle exploite fortement des signaux liés au genre.

Remarque :
le mot “convergence” est préférable à “cohésion”, car il indique que deux méthodes différentes identifient des indices similaires.







prompt pour le BERT : 


- Maintenant fais exactement pour le BERT 
- sauvegarde dans BERT/presentation
- les images desn BERT/presentation/images

- le dataset laisse le comme TF-IDF.



- les chunks on prent 64 homme 32 femme 


- entrainement 1 : fine tuning 
- entainement 2: linear probing


- evaluation : matrice de confusion 
- les hyperparamètres : 


BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE	0.00002
AdamW

- pour le dataset de l'entrainement dire que :  on prends 
NUM_CHUNKS_HOMME = 64
NUM_CHUNKS_FEMME = 32
par documents car y a beaucoup plus de documents femme que homme 




- instead of having integrated gradients and LRP we have now SHAP and LIME.







- mets la matrice de confusion de fine tuning a coté du plot de l'entrainemt de loss 
- dans la slide 12, enlève enlève la figure placeholder de matrice confusion car c'est dupliqué 
- tu mets la matrice de confusion de linear probing 
- enleve le plot de validation 
- enleve le text de le coeur de la méthode ...
- puis enleve la matrice de la confusion dans la slide juste après car elle est répétée ! 

- présente aussi l'architecture du réseau 


- juste avant les perturbations tu mets les attentions weights 




- enleve les perturbations
- tu dois mettre les explications locales  pour shap et lime 





