import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from textblob import TextBlob
from scipy.stats import ttest_ind
import seaborn as sns

def calculate_digit_count(document_vector, vocab_map):
    """
    Calcule le nombre total de chiffres dans un document.
    
    Arguments:
    - document_vector : numpy array, vecteur de fréquence d'un document
    - vocab_map : liste des mots du vocabulaire
    
    Retourne :
    - digit_count : int, nombre total de chiffres dans le document
    """
    digit_count = 0
    
    for idx, freq in enumerate(document_vector):
        word = vocab_map[idx]
        if any(char.isdigit() for char in word):  # Vérifie si le mot contient un chiffre
            digit_count += freq
    
    return digit_count

def analyze_digit_count(data_train, labels_train, vocab_map):
    """
    Calcule le nombre total de chiffres pour chaque document, affiche les distributions par classe
    et effectue un test de Mann-Whitney U pour évaluer si c'est une caractéristique intéressante.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    - labels_train : numpy array, labels de classe pour chaque document
    - vocab_map : liste des mots du vocabulaire
    
    Retourne :
    - p_value : float, p-valeur du test de Mann-Whitney U
    """
    digit_counts = []
    for document_vector in data_train:
        digit_count = calculate_digit_count(document_vector, vocab_map)
        digit_counts.append(digit_count)
    
    data = pd.DataFrame({'digit_count': digit_counts, 'label': labels_train})
    
    # Séparer le nombre de chiffres par classe
    digit_count_class_0 = data[data['label'] == 0]['digit_count']
    digit_count_class_1 = data[data['label'] == 1]['digit_count']
    
    # Afficher les distributions du nombre de chiffres par classe
    plt.figure(figsize=(12, 6))
    plt.hist(digit_count_class_0, bins=20, alpha=0.5, label='Classe 0', color='blue')
    plt.hist(digit_count_class_1, bins=20, alpha=0.5, label='Classe 1', color='orange')
    plt.xlabel('Nombre de chiffres')
    plt.ylabel('Fréquence')
    plt.title("Distribution du nombre de chiffres par classe")
    plt.legend()
    plt.show()
    
    # Effectuer le test de Mann-Whitney U
    stat, p_value = mannwhitneyu(digit_count_class_0, digit_count_class_1)
    
    print("P-value du test de Mann-Whitney U sur le nombre de chiffres :", p_value)
    if p_value < 0.05:
        print("Les distributions du nombre de chiffres sont significativement différentes entre les classes.")
        print("Cela pourrait être une caractéristique intéressante.")
    else:
        print("Les distributions du nombre de chiffres ne sont pas significativement différentes entre les classes.")
        print("Cela pourrait ne pas être une caractéristique intéressante.")
    
    return p_value



def reconstruct_text(document_vector, vocab_map):
    words = []
    for idx, freq in enumerate(document_vector):
        word = vocab_map[idx]
        words.extend([word] * int(freq))
    return " ".join(words)

def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def add_sentiment_feature(data_train, labels_train, vocab_map):
    sentiments = []
    for document_vector in data_train:
        text = reconstruct_text(document_vector, vocab_map)
        sentiment = calculate_sentiment(text)
        sentiments.append(sentiment)
    
    data = pd.DataFrame({'sentiment': sentiments, 'label': labels_train})
    return data

def test_sentiment_feature_ttest(data):
    sentiment_class_0 = data[data['label'] == 0]['sentiment']
    sentiment_class_1 = data[data['label'] == 1]['sentiment']
    # Test t de Student pour deux échantillons indépendants
    stat, p_value = ttest_ind(sentiment_class_0, sentiment_class_1, equal_var=False)  # "equal_var=False" pour tailles de classe différentes
    return p_value

def plot_sentiment_distributions(data):
    plt.figure(figsize=(12, 6))
    sentiment_class_0 = data[data['label'] == 0]['sentiment']
    sentiment_class_1 = data[data['label'] == 1]['sentiment']
    
    plt.hist(sentiment_class_0, bins=30, alpha=0.6, label='Classe 0', color='blue', density=False)
    plt.hist(sentiment_class_1, bins=30, alpha=0.6, label='Classe 1', color='orange', density=False)
    
    plt.xlabel("Score de sentiment")
    plt.ylabel("Fréquence")
    plt.title("Distribution des scores de sentiment par classe")
    plt.legend()
    plt.show()


def compute_text_lengths(data_train):
    """
    Calcule la longueur de chaque texte en comptant le nombre de termes non nuls dans chaque document.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    
    Retourne :
    - text_lengths : numpy array, longueurs de chaque texte
    """
    # Calculer la longueur de chaque texte en comptant les termes non nuls
    text_lengths = np.sum(data_train > 0, axis=1)
    return text_lengths

def analyze_text_length_distribution(data_train, labels_train):
    """
    Analyse et compare les distributions des longueurs de textes entre deux classes.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    - labels_train : numpy array, labels de classe pour chaque document
    
    Affiche les distributions et retourne le résultat du test statistique.
    """
    # Calculer les longueurs de texte pour chaque document
    text_lengths = compute_text_lengths(data_train)
    
    # Séparer les longueurs de texte par classe
    lengths_class_0 = text_lengths[labels_train == 0]
    lengths_class_1 = text_lengths[labels_train == 1]
    
    # Visualiser les distributions des longueurs de texte pour chaque classe
    plt.figure(figsize=(12, 6))
    sns.histplot(lengths_class_0, kde=True, color='blue', label='Classe 0', stat="count")
    sns.histplot(lengths_class_1, kde=True, color='orange', label='Classe 1', stat="count")
    plt.xlabel("Longueur du texte (nombre de mots)")
    plt.ylabel("Nombre de textes")
    plt.title("Distribution des longueurs de texte par classe")
    plt.legend()
    plt.show()
    
    # Effectuer le test t de Student pour comparer les distributions
    stat, p_value = ttest_ind(lengths_class_0, lengths_class_1, equal_var=False)
    print(f"Statistique de test : {stat}, p-value : {p_value}")
    
    # Interpréter le résultat
    if p_value < 0.05:
        print("Les distributions des longueurs de texte sont significativement différentes entre les classes (p < 0.05).")
        print("Cela suggère que la longueur du texte pourrait être une caractéristique utile pour la classification.")
    else:
        print("Les distributions des longueurs de texte ne sont pas significativement différentes entre les classes (p >= 0.05).")
        print("Cela suggère que la longueur du texte pourrait ne pas être une caractéristique utile pour la classification.")



def compute_unique_word_counts(data_train):
    """
    Calcule le nombre de mots distincts dans chaque texte.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    
    Retourne :
    - unique_word_counts : numpy array, nombre de mots distincts pour chaque texte
    """
    # Calculer le nombre de mots distincts (non nuls) dans chaque document
    unique_word_counts = np.sum(data_train > 0, axis=1)
    return unique_word_counts

def analyze_unique_word_count_distribution(data_train, labels_train):
    """
    Analyse et compare les distributions du nombre de mots distincts par texte entre deux classes.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    - labels_train : numpy array, labels de classe pour chaque document
    
    Affiche les distributions et retourne le résultat du test statistique.
    """
    # Calculer le nombre de mots distincts pour chaque texte
    unique_word_counts = compute_unique_word_counts(data_train)
    
    # Séparer les nombres de mots distincts par classe
    unique_counts_class_0 = unique_word_counts[labels_train == 0]
    unique_counts_class_1 = unique_word_counts[labels_train == 1]
    
    # Visualiser les distributions du nombre de mots distincts par classe
    plt.figure(figsize=(12, 6))
    sns.histplot(unique_counts_class_0, kde=True, color='blue', label='Classe 0', stat="count")
    sns.histplot(unique_counts_class_1, kde=True, color='orange', label='Classe 1', stat="count")
    plt.xlabel("Nombre de mots distincts par texte")
    plt.ylabel("Nombre de textes")
    plt.title("Distribution du nombre de mots distincts par texte pour chaque classe")
    plt.legend()
    plt.show()
    
    # Effectuer le test t de Student pour comparer les distributions
    stat, p_value = ttest_ind(unique_counts_class_0, unique_counts_class_1, equal_var=False)
    print(f"Statistique de test t : {stat}, p-value : {p_value}")
    
    # Interpréter le résultat
    if p_value < 0.05:
        print("Les distributions du nombre de mots distincts par texte sont significativement différentes entre les classes (p < 0.05).")
        print("Cela suggère que le nombre de mots distincts pourrait être une caractéristique utile pour la classification.")
    else:
        print("Les distributions du nombre de mots distincts par texte ne sont pas significativement différentes entre les classes (p >= 0.05).")
        print("Cela suggère que le nombre de mots distincts pourrait ne pas être une caractéristique utile pour la classification.")

