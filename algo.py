import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Fonction sigmoïde."""
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        """Initialiser les poids et le biais."""
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def forward_propagation(self, X):
        """Propagation vers l'avant : calculer la prédiction."""
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return y_hat

    def compute_cost(self, y_hat, y):
        """Calculer la fonction de coût (log-loss)."""
        m = y.shape[0]
        cost = - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return cost

    def backward_propagation(self, X, y_hat, y):
        """Calculer les gradients des paramètres (rétropropagation)."""
        m = X.shape[0]
        dz = y_hat - y
        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        return dw, db

    def update_parameters(self, dw, db):
        """Mettre à jour les poids et le biais."""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        """Entraîner le modèle de régression logistique."""
        # Initialiser les paramètres
        num_features = X.shape[1]
        self.initialize_parameters(num_features)

        # Convertir y en un vecteur colonne si nécessaire
        y = y.reshape(-1, 1)

        # Gradient descent
        for i in range(self.num_iterations):
            # Forward propagation
            y_hat = self.forward_propagation(X)

            # Calculer la fonction de coût
            cost = self.compute_cost(y_hat, y)

            # Backward propagation
            dw, db = self.backward_propagation(X, y_hat, y)

            # Mettre à jour les paramètres
            self.update_parameters(dw, db)

            # Afficher le coût tous les 100 itérations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")

    def predict(self, X):
        """Prédire les étiquettes pour un ensemble de données."""
        y_hat = self.forward_propagation(X)
        return np.where(y_hat >= 0.5, 1, 0)





import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, f1_score, classification_report

def random_search_naive_bayes(data_train, labels_train, n_iter=10):
    """
    Effectue une recherche aléatoire pour optimiser les hyperparamètres d'un modèle Naïve Bayes multinomial.

    Arguments:
    - data_train : numpy array, matrice terme-document pour l'entraînement
    - labels_train : numpy array, labels de classe pour chaque document d'entraînement
    - n_iter : int, nombre d'itérations pour la recherche aléatoire
    
    Retourne :
    - best_model : modèle entraîné avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    - search_results : dataframe des résultats de la recherche
    """
    # Définir le modèle Naïve Bayes multinomial
    nb = MultinomialNB()

    # Définir les paramètres à rechercher
    param_distributions = {
        'alpha': np.linspace(0.01, 2, 50),  # Paramètre de lissage
        'fit_prior': [True, False]         # Utiliser ou non les probabilités a priori
    }

    # Métrique d'évaluation
    scoring = make_scorer(f1_score, average='macro')

    # Définir une validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Recherche aléatoire
    random_search = RandomizedSearchCV(
        estimator=nb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Effectuer la recherche
    random_search.fit(data_train, labels_train)

    # Meilleurs paramètres et modèle
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Convertir les résultats en DataFrame
    search_results = pd.DataFrame(random_search.cv_results_).sort_values(by='rank_test_score')

    print("Meilleurs paramètres :", best_params)
    print("Meilleur F1-score (validation) :", random_search.best_score_)
    
    return best_model, best_params, search_results



import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

def bayesian_search_naive_bayes(data_train, labels_train, n_iter=50):
    """
    Effectue une recherche bayésienne pour optimiser les hyperparamètres d'un modèle Naïve Bayes multinomial.

    Arguments:
    - data_train : numpy array, matrice terme-document pour l'entraînement
    - labels_train : numpy array, labels de classe pour chaque document d'entraînement
    - n_iter : int, nombre d'itérations pour la recherche bayésienne
    
    Retourne :
    - best_model : modèle entraîné avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    - search_results : dataframe des résultats de la recherche
    """
    # Définir le modèle Naïve Bayes multinomial
    nb = MultinomialNB()

    # Définir l'espace de recherche pour les hyperparamètres
    search_space = {
        'alpha': Real(0.01, 2.0, prior='uniform'),  # Paramètre de lissage
        'fit_prior': Categorical([True, False])    # Utiliser ou non les probabilités a priori
    }

    # Métrique d'évaluation
    scoring = make_scorer(f1_score, average='macro')

    # Définir une validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Recherche bayésienne
    bayes_search = BayesSearchCV(
        estimator=nb,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Effectuer la recherche
    bayes_search.fit(data_train, labels_train)

    # Meilleurs paramètres et modèle
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    # Convertir les résultats en DataFrame
    search_results = pd.DataFrame(bayes_search.cv_results_).sort_values(by='rank_test_score')

    print("Meilleurs paramètres :", best_params)
    print("Meilleur F1-score (validation) :", bayes_search.best_score_)

    return best_model, best_params, search_results

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV

def bayesian_optimization_logistic_regression(data_train, labels_train):
    """
    Effectue une recherche bayésienne pour optimiser les hyperparamètres d'un modèle de régression logistique.
    
    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe pour chaque échantillon
    
    Retourne :
    - best_model : le modèle de régression logistique avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle de régression logistique
    logistic_regression = LogisticRegression(max_iter=1000, solver='lbfgs')

    # Définir l'espace de recherche pour les hyperparamètres
    search_space = {
        'C': (1e-6, 1000.0, 'log-uniform'),  # Paramètre de régularisation L2
        'solver': ['lbfgs', 'liblinear'],   # Solvers disponibles
        'penalty': ['l2'],                  # Régularisation L2
    }

    # Configurer la recherche Bayesienne
    bayes_search = BayesSearchCV(
        estimator=logistic_regression,
        search_spaces=search_space,
        scoring='f1_macro',  # Métrique à optimiser
        n_iter=30,           # Nombre d'itérations de recherche
        cv=5,                # Validation croisée à 5 plis
        verbose=1,
        n_jobs=-1            # Utiliser tous les cœurs disponibles
    )

    # Lancer l'optimisation
    bayes_search.fit(data_train, labels_train)

    # Récupérer les meilleurs paramètres
    best_params = bayes_search.best_params_
    best_model = bayes_search.best_estimator_

    print("Meilleurs paramètres :", best_params)
    print("Meilleure performance (F1 macro) :", bayes_search.best_score_)

    return best_model, best_params


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

def random_search_logistic_regression(data_train, labels_train, n_iter=30):
    """
    Effectue une recherche aléatoire pour optimiser les hyperparamètres d'un modèle de régression logistique.
    
    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe pour chaque échantillon
    - n_iter : int, nombre d'itérations pour la recherche aléatoire
    
    Retourne :
    - best_model : le modèle de régression logistique avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle de régression logistique
    logistic_regression = LogisticRegression(max_iter=1000, solver='lbfgs')

    # Définir les paramètres à rechercher
    param_distributions = {
        'C': np.logspace(-6, 3, 50),  # Paramètre de régularisation L2
        'solver': ['lbfgs', 'liblinear'],  # Solvers disponibles
        'penalty': ['l2'],  # Régularisation L2 uniquement
    }

    # Configurer la validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Métrique d'évaluation
    scoring = make_scorer(f1_score, average='macro')

    # Configurer la recherche aléatoire
    random_search = RandomizedSearchCV(
        estimator=logistic_regression,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Utiliser tous les cœurs disponibles
    )

    # Lancer la recherche
    random_search.fit(data_train, labels_train)

    # Meilleurs paramètres et modèle
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Meilleurs paramètres :", best_params)
    print("Meilleure performance (F1 macro) :", random_search.best_score_)

    return best_model, best_params


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

def bayesian_optimization_linear_svm(data_train, labels_train):
    """
    Optimise le paramètre C pour un SVM linéaire en utilisant une recherche bayésienne.

    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe
    
    Retourne :
    - best_model : modèle entraîné avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle SVM linéaire
    linear_svm = SVC(kernel='linear', random_state=42)

    # Définir l'espace de recherche pour \( C \)
    search_space = {
        'C': (1e-3, 1e3, 'log-uniform')  # Recherche sur une échelle logarithmique
    }

    # Configurer la validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Recherche bayésienne
    bayes_search = BayesSearchCV(
        estimator=linear_svm,
        search_spaces=search_space,
        scoring='f1_macro',  # Optimisation pour le F1-score macro
        n_iter=30,           # Nombre d'itérations
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Exécuter la recherche
    bayes_search.fit(data_train, labels_train)

    # Extraire le meilleur modèle et les paramètres
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    print("Meilleurs paramètres trouvés :", best_params)
    print("Meilleur F1-score (validation) :", bayes_search.best_score_)

    return best_model, best_params


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

def random_search_linear_svm(data_train, labels_train, n_iter=20):
    """
    Optimise le paramètre C pour un SVM linéaire en utilisant une recherche aléatoire.

    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe
    - n_iter : int, nombre d'itérations pour la recherche aléatoire
    
    Retourne :
    - best_model : modèle entraîné avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle SVM linéaire
    linear_svm = SVC(kernel='linear', random_state=42)

    # Définir les paramètres pour la recherche
    param_distributions = {
        'C': np.logspace(-3, 3, 50)  # Recherche de \( C \) sur une échelle logarithmique
    }

    # Configurer la validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Configurer RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=linear_svm,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=make_scorer(f1_score, average='macro'),  # Optimiser le F1-score macro
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Exécuter la recherche
    random_search.fit(data_train, labels_train)

    # Extraire le meilleur modèle et les paramètres
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Meilleurs paramètres trouvés :", best_params)
    print("Meilleur F1-score (validation) :", random_search.best_score_)

    return best_model, best_params

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, f1_score

def bayesian_optimization_random_forest(data_train, labels_train):
    """
    Effectue une recherche bayésienne pour optimiser les hyperparamètres d'un modèle Random Forest.

    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe

    Retourne :
    - best_model : le modèle Random Forest avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle Random Forest
    random_forest = RandomForestClassifier(random_state=42)

    # Définir l'espace de recherche pour les hyperparamètres
    search_space = {
        'n_estimators': (50, 300),          # Nombre d'arbres dans la forêt
        'max_depth': (3, 20),              # Profondeur maximale des arbres
        'min_samples_split': (2, 10),      # Nombre minimum d'échantillons pour diviser un nœud
        'min_samples_leaf': (1, 5),        # Nombre minimum d'échantillons dans une feuille
        'max_features': ['sqrt', 'log2', None]  # Nombre maximum de caractéristiques pour chaque split
    }

    # Configurer la recherche bayésienne
    bayes_search = BayesSearchCV(
        estimator=random_forest,
        search_spaces=search_space,
        scoring=make_scorer(f1_score, average='macro'),  # Métrique à optimiser
        n_iter=30,                # Nombre d'itérations de recherche
        cv=5,                     # Validation croisée à 5 plis
        verbose=2,
        n_jobs=-1,                # Utiliser tous les cœurs disponibles
        random_state=42
    )

    # Lancer l'optimisation
    bayes_search.fit(data_train, labels_train)

    # Récupérer les meilleurs paramètres
    best_params = bayes_search.best_params_
    best_model = bayes_search.best_estimator_

    print("Meilleurs paramètres :", best_params)
    print("Meilleure performance (F1 macro) :", bayes_search.best_score_)

    return best_model, best_params

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

def random_search_random_forest(data_train, labels_train, n_iter=30):
    """
    Effectue une recherche aléatoire pour optimiser les hyperparamètres d'un modèle Random Forest.

    Arguments:
    - data_train : numpy array, matrice des caractéristiques
    - labels_train : numpy array, labels de classe
    - n_iter : int, nombre d'itérations pour la recherche aléatoire

    Retourne :
    - best_model : le modèle Random Forest avec les meilleurs paramètres
    - best_params : dict, les meilleurs paramètres trouvés
    """
    # Définir le modèle Random Forest
    random_forest = RandomForestClassifier(random_state=42)

    # Définir les paramètres à rechercher
    param_distributions = {
        'n_estimators': np.arange(50, 301, 50),      # Nombre d'arbres dans la forêt
        'max_depth': np.arange(3, 21, 1),           # Profondeur maximale des arbres
        'min_samples_split': np.arange(2, 11),      # Nombre minimum d'échantillons pour diviser un nœud
        'min_samples_leaf': np.arange(1, 6),        # Nombre minimum d'échantillons dans une feuille
        'max_features': ['sqrt', 'log2', None]      # Nombre maximum de caractéristiques pour chaque split
    }

    # Configurer la recherche aléatoire
    random_search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=make_scorer(f1_score, average='macro'),  # Métrique à optimiser
        cv=5,                     # Validation croisée à 5 plis
        verbose=2,
        random_state=42,
        n_jobs=-1                 # Utiliser tous les cœurs disponibles
    )

    # Lancer l'optimisation
    random_search.fit(data_train, labels_train)

    # Récupérer les meilleurs paramètres
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    print("Meilleurs paramètres :", best_params)
    print("Meilleure performance (F1 macro) :", random_search.best_score_)

    return best_model, best_params


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def evaluate_models_with_pipelines(data, labels):
    """
    Évalue plusieurs modèles (Régression Logistique, SVM, Random Forest, Naïve Bayes)
    en utilisant des pipelines, trace les performances par fold et les courbes ROC.
    """
    # Création des pipelines avec prétraitement
    pipelines = {
        "Naïve Bayes": Pipeline([
            ('model', MultinomialNB(alpha=0.01, fit_prior=True))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('model', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42))
        ]),
        "SVM (linéaire)": Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('model', SVC(C=0.75, kernel='linear', probability=True, max_iter=5000))
        ]),
        "Régression Logistique": Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('model', LogisticRegression(C=0.1037, penalty='l2', solver='liblinear', max_iter=5000))
        ]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, pipeline in pipelines.items():
        train_f1_scores = []
        test_f1_scores = []
        train_accuracies = []
        test_accuracies = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Entraîner le modèle via le pipeline
            pipeline.fit(X_train, y_train)

            # Prédictions
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            # Calcul des scores
            train_f1_scores.append(f1_score(y_train, y_train_pred, average='macro'))
            test_f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
            train_accuracies.append(accuracy_score(y_train, y_train_pred))
            test_accuracies.append(accuracy_score(y_test, y_test_pred))

            # Calculer les probabilités pour la courbe ROC
            y_proba = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline.named_steps['model'], "predict_proba")
                else pipeline.decision_function(X_test)
            )
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

        # Afficher les performances par fold
        plot_fold_performance(model_name, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores)

        # Courbe ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plot_roc_curve(model_name, mean_fpr, mean_tpr)

def plot_fold_performance(model_name, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores):
    """
    Trace un diagramme à lignes brisées pour les performances par fold.
    """
    plt.figure(figsize=(10, 6))
    folds = np.arange(1, len(train_accuracies) + 1)
    plt.plot(folds, train_accuracies, marker='o', label="Train Accuracy", color='blue')
    plt.plot(folds, test_accuracies, marker='s', label="Test Accuracy", color='orange')
    plt.plot(folds, train_f1_scores, marker='^', label="Train F1 Score", color='green')
    plt.plot(folds, test_f1_scores, marker='x', label="Test F1 Score", color='red')

    plt.title(f"Performance des folds pour {model_name}")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model_name, fpr, tpr):
    """
    Trace la courbe ROC pour un modèle donné.
    """
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC pour {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_and_predict_multinomial_nb(data_train, labels_train, data_test, alpha=0.01, fit_prior=True, output_file="predictions.csv"):
    """
    Entraîne un modèle Multinomial Naive Bayes avec les paramètres optimaux et prédit les classes pour le jeu de données test.

    Arguments:
    - data_train : numpy array, matrice des données d'entraînement
    - labels_train : numpy array, labels associés aux données d'entraînement
    - data_test : numpy array, matrice des données de test
    - alpha : float, paramètre de lissage pour le modèle Naive Bayes
    - fit_prior : bool, utiliser ou non les probabilités a priori
    - output_file : str, nom du fichier de sortie pour les prédictions

    Retourne :
    - predictions : numpy array, prédictions pour le jeu de données test
    """
    # Initialiser le modèle avec les paramètres optimaux
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    # Entraîner le modèle
    model.fit(data_train, labels_train)

    # Faire les prédictions sur le jeu de données test
    predictions = model.predict(data_test)

    # Sauvegarder les prédictions dans un fichier CSV
    results = pd.DataFrame({"ID": np.arange(len(predictions)), "label": predictions})
    results.to_csv(output_file, index=False)

    print(f"Prédictions enregistrées dans le fichier : {output_file}")
    return predictions

