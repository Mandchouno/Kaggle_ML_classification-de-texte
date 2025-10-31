`*`*La version en français suit la version en anglais*`*`  
`*`*French version follows*`*`

# Text Classification – Kaggle Competition (IFT3395 / IFT6390)

***Date :** Fall 2024*

## Description

This project was developed as part of the **IFT3395 / IFT6390 – Machine Learning** course at the **Université de Montréal**.  
It aims to classify textual documents into two predefined categories using a machine learning pipeline optimized for **imbalanced datasets**.

The workflow includes **text preprocessing**, **feature engineering**, **model comparison**, and **hyperparameter optimization**, with the primary performance metric being the **macro F1-score**.

The project was submitted as part of the **Kaggle Competition** organized for the course.

---

## Project structure

Kaggle_Text_Classification/
│
├── algo.py                     # Implementation of main algorithms
├── baseline Methode.py         # Baseline model (benchmark)
├── feature_creation.py         # Feature engineering (sentiment, word count, etc.)
├── visualisation.py            # Data visualization and ROC/F1 analysis
├── code final.ipynb            # Final Jupyter notebook with experiments
├── Kaggle_competition___IFT3395_IFT6390___2024__Copy_.pdf   # Full project report
└── README.md                   # Documentation file

---

## Methodology

* **Data preprocessing**  
  * Lemmatization to reduce vocabulary size and maintain semantic meaning.  
  * Removal of rare terms (appearing in fewer than 3 documents).  
  * Retention of stopwords for stylistic variance detection.  
  * Feature selection using the **Mann–Whitney U test** for non-normal distributions.

* **Feature engineering**  
  * Sentiment analysis (polarity and subjectivity) via *TextBlob*.  
  * Text length features: number of words, unique words, digits.  
  * Feature significance tested using *t-tests* and *p-values*.

* **Algorithms tested**  
  * Logistic Regression  
  * Multinomial Naïve Bayes  
  * Support Vector Machine (SVM)  
  * Random Forest  

* **Hyperparameter optimization**  
  * Combination of *random search* and *Bayesian optimization*.  
  * Stratified cross-validation to handle class imbalance.

---

## Results

| Model | Macro F1-score | Notes |
|:------|:---------------:|------:|
| Logistic Regression | 0.6946 | Stable, interpretable baseline |
| SVM | 0.6832 | Good margin separation, slower runtime |
| Random Forest | 0.6754 | Nonlinear modeling, higher complexity |
| **Multinomial Naïve Bayes** | **0.7491** | Best model after feature optimization |

**Key insights:**
* The **Naïve Bayes** model achieved the best performance (Macro F1 = 0.7491).  
* Feature selection based on the Mann–Whitney U test significantly improved accuracy.  
* Validation curves and ROC analysis confirm strong generalization (AUC = 0.81).  

---

## Technologies used

* **Python 3.8+**
* **Libraries:**  
  * `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `textblob`, `scipy`
* **Environment:** Jupyter Notebook
* **Platform:** Kaggle / local execution

---

## How to run

## How to run

1. Clone or download the repository:
   ```bash
   git clone https://github.com/<your-username>/Kaggle_Text_Classification.git
   cd Kaggle_Text_Classification
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3.	Run the baseline or final model:
  python "baseline Methode.py"
  python algo.py
4.	To explore the complete pipeline:
  jupyter notebook "code final.ipynb"

---

## Credits

This project was developed by Cédric Kamdem and Mandi Téo Vigier
as part of the Kaggle competition for IFT3395 / IFT6390 – Machine Learning at the Université de Montréal.

<br>
# Text Classification – Kaggle Competition (IFT3395 / IFT6390)

***Date :** Automne 2024*

## Description

Ce projet a été réalisé dans le cadre du cours IFT3395 / IFT6390 – Apprentissage Automatique à l’Université de Montréal.
L’objectif est de classer des documents textuels en deux catégories prédéfinies à l’aide d’un pipeline de machine learning optimisé pour les jeux de données déséquilibrés.

Le pipeline comprend le prétraitement des données, la création de caractéristiques, la comparaison des modèles, et l’optimisation des hyperparamètres, avec le macro F1-score comme métrique principale.

Le projet a été soumis dans le cadre de la compétition Kaggle du cours.

⸻

## Structure du projet

Kaggle_Text_Classification/
│
├── algo.py                     # Implémentation des algorithmes principaux
├── baseline Methode.py         # Modèle de référence (benchmark)
├── feature_creation.py         # Création de caractéristiques (sentiment, longueur, etc.)
├── visualisation.py            # Visualisation et analyse des performances
├── code final.ipynb            # Notebook final des expériences
├── Kaggle_competition___IFT3395_IFT6390___2024__Copy_.pdf   # Rapport complet
└── README.md                   # Documentation

---

## Méthodologie
- Prétraitement des données
    	•	Lemmatisation pour réduire le vocabulaire tout en préservant le sens.
    	•	Filtrage des mots apparaissant dans moins de 3 documents.
    	•	Conservation des stopwords pour capturer les variations de style.
    	•	Sélection de variables avec le test Mann–Whitney U.
- Création des caractéristiques
    	•	Analyse de sentiment (polarité et subjectivité) avec TextBlob.
    	•	Longueur du texte (nombre total et distinct de mots, chiffres).
    	•	Vérification de la significativité par tests t et p-values.
- Algorithmes testés
    	•	Régression logistique
    	•	Naïve Bayes multinomial
    	•	SVM
    	•	Forêt aléatoire
- Optimisation des hyperparamètres
    	•	Recherche aléatoire combinée à une optimisation bayésienne.
    	•	Validation croisée stratifiée pour les classes déséquilibrées.

--- 

## Technologies utilisées
	•	Python 3.8+
	•	Bibliothèques :
	•	scikit-learn, numpy, pandas, matplotlib, seaborn, textblob, scipy
	•	Environnement : Jupyter Notebook
	•	Plateforme : Kaggle / locale

---

# Execution 

1. Cloner ou télécharger le dépôt:
   ```bash
   git clone https://github.com/<your-username>/Kaggle_Text_Classification.git
   cd Kaggle_Text_Classification
2. Installer les dépendances:
   ```bash
   pip install -r requirements.txt
3.	Exécuter les scripts:
  python "baseline Methode.py"
  python algo.py
4.	Lancer le notebook complet:
  jupyter notebook "code final.ipynb"

---

## Crédits

Projet réalisé par Cédric Kamdem et Mandi Téo Vigier
dans le cadre de la compétition Kaggle du cours IFT3395 / IFT6390 – Apprentissage Automatique à l’Université de Montréal.
