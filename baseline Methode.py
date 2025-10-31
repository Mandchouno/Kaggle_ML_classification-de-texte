import numpy as np
import pandas as pd

# Classe Naive Bayes
class NaiveBayesClassifier:
    def fit(self, X, y):
        # Calculer les probabilités a priori pour chaque classe
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.class_prior = class_counts / len(y)

        # Calculer la probabilité conditionnelle de chaque mot pour chaque classe
        self.word_counts = {}
        self.word_prob = {}
        
        for cls in self.classes:
            X_class = X[y == cls]
            self.word_counts[cls] = np.sum(X_class, axis=0) + 1  # Lissage de Laplace
            self.word_prob[cls] = self.word_counts[cls] / np.sum(self.word_counts[cls])

    def predict(self, X):
        log_likelihood = []
        
        for cls in self.classes:
            # Calculer la log-vraisemblance pour chaque classe
            log_prior = np.log(self.class_prior[cls])
            log_conditional = X @ np.log(self.word_prob[cls].T)  # produit matriciel
            log_likelihood_cls = log_prior + log_conditional
            log_likelihood.append(log_likelihood_cls)
        
        # Comparer la vraisemblance pour chaque classe et choisir la classe avec la probabilité maximale
        log_likelihood = np.array(log_likelihood)
        predictions = np.argmax(log_likelihood, axis=0)
        return predictions

# Charger les données d'entraînement
X_train = np.load('data_train.npy', allow_pickle=True)
labels_df = pd.read_csv('label_train.csv')
y_train = labels_df['label'].values
X_test = np.load('data_test.npy', allow_pickle=True)





# Initialiser et entraîner le modèle Naive Bayes avec les données sous-échantillonnées
nb_model = NaiveBayesClassifier()
nb_model.fit(X_train, y_train)

# Faire les prédictions

y_pred_test =nb_model.predict(X_test)

# Sauvegarder les prédictions de l'ensemble de test dans un fichier CSV
df_predictions = pd.DataFrame({
    'ID': range(len(y_pred_test)),  # Remplace par les IDs de tes documents si nécessaire
    'label': y_pred_test
})
df_predictions.to_csv('predictions_naive_bayes.csv', index=False)