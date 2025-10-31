import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.sparse import csr_matrix
from wordcloud import WordCloud
import re
from collections import Counter


def distribution_mot(data_train, labels_train, vocab_map,log=False ):
    # Somme des occurrences de chaque terme
    term_frequencies = np.sum(data_train, axis=0)

    if log:
        term_frequencies = np.log1p(term_frequencies) 
    

    # Trier les termes par fréquence décroissante
    sorted_indices = np.argsort(term_frequencies)[::-1]

    # Associer les mots aux indices triés
    frequent_words = [vocab_map[i] for i in sorted_indices[:25]]

    # Afficher les 10 mots les plus fréquents avec leurs fréquences
    print("Les 25 mots les plus fréquents et leurs fréquences :")
    for word, freq in zip(frequent_words, term_frequencies[sorted_indices[:25]]):
        print(f"{word}: {freq}")

    # Visualisation de la distribution des fréquences des termes
    plt.figure(figsize=(10, 6))
    plt.hist(term_frequencies, bins=50, log=True)
    plt.title('Distribution des fréquences des termes')
    plt.xlabel('Fréquence des termes')
    plt.ylabel('Nombre de termes')
    plt.show()
    return None
    


def loi_de_Zipf(data_train, labels_train, vocab_map):

    term_frequencies = np.sum(data_train, axis=0)
    # Trier les termes par fréquence décroissante
    sorted_term_frequencies = np.sort(term_frequencies)[::-1]

    # Afficher quelques statistiques sur les fréquences des termes
    print(f"Fréquence du terme le plus fréquent : {sorted_term_frequencies[0]}")
    print(f"Fréquence moyenne des termes : {np.mean(sorted_term_frequencies):.2f}")


    # Tracer la distribution des fréquences des termes (log-log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(sorted_term_frequencies) + 1), sorted_term_frequencies)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Distribution des fréquences des termes (loi de Zipf)')
    plt.xlabel('Rang des termes')
    plt.ylabel('Fréquence des termes')
    plt.show()
    return None
 

def nuage_termes_frequent(data_train, labels_train, vocab_map): 
    # Somme des occurrences de chaque terme
    term_frequencies = np.sum(data_train, axis=0)
    # Créer un dictionnaire des fréquences des mots avec vocab_map
    word_dict = {vocab_map[i]: term_frequencies[i] for i in range(len(term_frequencies))}

    # Générer un nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)

    # Afficher le nuage de mots
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nuage de mots des termes les plus fréquents")
    plt.show()
    return None


def distribution_cumule_frequence_termes(data_train, labels_train, vocab_map):
    # Somme des occurrences de chaque terme
    term_frequencies = np.sum(data_train, axis=0)

    # Tri des fréquences des termes par ordre décroissant
    sorted_frequencies = np.sort(term_frequencies)[::-1]

    # Distribution cumulée
    cumulative_frequencies = np.cumsum(sorted_frequencies) / np.sum(sorted_frequencies)

    # Tracer la distribution cumulée
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_frequencies)
    plt.title('Distribution cumulée des fréquences des termes')
    plt.xlabel('Nombre de termes')
    plt.ylabel('Proportion cumulée des occurrences')
    plt.show()


    # Afficher combien de termes couvrent 50% des occurrences
    terms_50_percent = np.argmax(cumulative_frequencies >= 0.50)
    print(f"Nombre de termes responsables de 50% des occurrences : {terms_50_percent}")

    # Afficher combien de termes couvrent 80% des occurrences
    terms_80_percent = np.argmax(cumulative_frequencies >= 0.80)
    print(f"Nombre de termes responsables de 80% des occurrences : {terms_80_percent}")

    # Afficher combien de termes couvrent 90% des occurrences
    terms_90_percent = np.argmax(cumulative_frequencies >= 0.90)
    print(f"Nombre de termes responsables de 90% des occurrences : {terms_90_percent}")

    # Afficher combien de termes couvrent 90% des occurrences
    terms_95_percent = np.argmax(cumulative_frequencies >= 0.95)
    print(f"Nombre de termes responsables de 95% des occurrences : {terms_95_percent}")

    # Afficher combien de termes couvrent 90% des occurrences
    terms_98_percent = np.argmax(cumulative_frequencies >= 0.98)
    print(f"Nombre de termes responsables de 98% des occurrences : {terms_98_percent}")




def distribution_longueur_document(data_train, labels_train, vocab_map):
    # Somme des termes dans chaque document
    terms_per_doc = np.sum(data_train, axis=1)

    # Visualisation de la distribution des longueurs des documents
    plt.figure(figsize=(10, 6))
    plt.hist(terms_per_doc, bins=50, log=True)
    plt.title('Distribution des longueurs des documents')
    plt.xlabel('Nombre de termes par document')
    plt.ylabel('Nombre de documents')
    plt.show()

    # Afficher quelques statistiques
    print(f"Nombre moyen de termes par document : {np.mean(terms_per_doc)}")
    print(f"Nombre minimum de termes dans un document : {np.min(terms_per_doc)}")
    print(f"Nombre maximum de termes dans un document : {np.max(terms_per_doc)}")


def distribution_classes(data_train, labels_train, vocab_map):
     # Calcul de la distribution des classes
    class_distribution = labels_train['label'].value_counts()

    # Visualisation de la distribution des classes
    plt.figure(figsize=(6, 4))
    class_distribution.plot(kind='bar', color=['skyblue', 'lightgreen'])
    plt.title('Distribution des classes cibles')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'exemples')
    plt.show()

    # Affichage des proportions
    print(f"Proportion de la classe 0 : {class_distribution[0] / len(labels_train):.2f}")
    print(f"Proportion de la classe 1 : {class_distribution[1] / len(labels_train):.2f}")


def distribution_document_par_classe(data_train, labels_train, vocab_map): 
    # Calculer la longueur de chaque document
    document_lengths = np.sum(data_train, axis=1)

    # Séparer les longueurs par classe
    document_lengths_class_0 = document_lengths[labels_train['label'] == 0]
    document_lengths_class_1 = document_lengths[labels_train['label'] == 1]

    # Statistiques sur la longueur des documents par classe
    print(f"Longueur moyenne des documents (classe 0) : {np.mean(document_lengths_class_0):.2f}")
    print(f"Longueur moyenne des documents (classe 1) : {np.mean(document_lengths_class_1):.2f}")

    # Visualisation de la distribution des longueurs par classe
    plt.figure(figsize=(10, 6))
    plt.hist(document_lengths_class_0, bins=50, alpha=0.5, label='Classe 0', log=True, color='skyblue')
    plt.hist(document_lengths_class_1, bins=50, alpha=0.5, label='Classe 1', log=True, color='lightgreen')
    plt.title('Distribution des longueurs des documents par classe')
    plt.xlabel('Nombre de termes')
    plt.ylabel('Nombre de documents')
    plt.legend()


def identificatino_termes_different(data_train, labels_train, vocab_map): 
    # Séparer les documents par classe
    class_0_docs = data_train[labels_train['label'] == 0, :]
    class_1_docs = data_train[labels_train['label'] == 1, :]

    # Fréquence des termes dans chaque classe
    term_freq_class_0 = np.sum(class_0_docs, axis=0)
    term_freq_class_1 = np.sum(class_1_docs, axis=0)

    # Termes distinctifs : différence de fréquence entre les classes
    distinctive_terms = term_freq_class_1 - term_freq_class_0

    # Trier les termes par différence de fréquence
    top_distinctive_indices = np.argsort(np.abs(distinctive_terms))[-10:]  # 10 termes les plus distinctifs

    # Afficher les termes distinctifs
    print("Les 10 termes les plus distinctifs entre les classes :")
    for idx in top_distinctive_indices:
        print(f"Terme : {vocab_map[idx]}, Différence de fréquence : {distinctive_terms[idx]:.2f}")


def identification_termes_differents_norm(data_train, labels_train, vocab_map):
    # Séparer les documents par classe
    class_0_docs = data_train[labels_train['label'] == 0, :]
    class_1_docs = data_train[labels_train['label'] == 1, :]

    # Fréquence des termes dans chaque classe
    term_freq_class_0 = np.sum(class_0_docs, axis=0)
    term_freq_class_1 = np.sum(class_1_docs, axis=0)

    # Calculer la longueur moyenne des documents dans chaque classe
    mean_length_class_0 = np.mean(np.sum(class_0_docs > 0, axis=1))
    mean_length_class_1 = np.mean(np.sum(class_1_docs > 0, axis=1))

    # Normaliser les fréquences des termes par le nombre de documents et la longueur moyenne des documents
    term_freq_class_0_normalized = term_freq_class_0 / (class_0_docs.shape[0] * mean_length_class_0)
    term_freq_class_1_normalized = term_freq_class_1 / (class_1_docs.shape[0] * mean_length_class_1)

    # Termes distinctifs : différence des fréquences normalisées entre les classes
    distinctive_terms = term_freq_class_1_normalized - term_freq_class_0_normalized

    # Trier les termes par différence de fréquence normalisée
    top_distinctive_indices = np.argsort(np.abs(distinctive_terms))[-10:]  # 10 termes les plus distinctifs

    # Afficher les termes distinctifs
    print("Les 10 termes les plus distinctifs entre les classes après normalisation :")
    for idx in top_distinctive_indices:
        print(f"Terme : {vocab_map[idx]}, Différence de fréquence normalisée : {distinctive_terms[idx]:.4f}")

    return top_distinctive_indices, distinctive_terms


def nuage_classe0(data_train, labels_train, vocab_map): 
    # Calculer la fréquence des termes pour la classe 0
    term_freq_class_0 = np.sum(data_train[labels_train['label'] == 0], axis=0)

    # Créer un dictionnaire de mots avec les fréquences associées
    word_freq_class_0 = {vocab_map[i]: term_freq_class_0[i] for i in range(len(vocab_map))}

    # Générer le nuage de mots pour la classe 0
    wordcloud_0 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_class_0)

    # Afficher le nuage de mots pour la classe 0
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_0, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots pour la classe 0')
plt.show()



def nuage_classe1(data_train, labels_train, vocab_map): 
    # Calculer la fréquence des termes pour la classe 1
    term_freq_class_1 = np.sum(data_train[labels_train['label'] == 1], axis=0)

    # Créer un dictionnaire de mots avec les fréquences associées
    word_freq_class_1 = {vocab_map[i]: term_freq_class_1[i] for i in range(len(vocab_map))}

    # Générer le nuage de mots pour la classe 1
    wordcloud_1 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_class_1)

    # Afficher le nuage de mots pour la classe 1
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_1, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots pour la classe 1')
    plt.show()





import numpy as np
from scipy.sparse import csr_matrix

def chi_square_test_for_word_with_weights(word_index, data_train, labels_train, class_weights):
    """
    Effectue un test du Chi² avec pondération pour un mot donné (indiqué par son index) pour voir s'il est lié à la classe.
    
    Arguments :
    - word_index : int, l'index du mot dans la matrice des données (vocab_map)
    - data_train : numpy array ou sparse matrix, matrice des termes (documents x termes)
    - labels_train : pandas DataFrame, étiquettes de classe pour chaque document
    - class_weights : dict, poids associés à chaque classe (ex: {0: 0.7, 1: 1.3})

    Retourne :
    - chi_square : float, score Chi² pondéré pour le mot
    """
    # Séparer les documents par classe
    class_0_docs = data_train[labels_train['label'] == 0, :]
    class_1_docs = data_train[labels_train['label'] == 1, :]

    # Fréquence observée du mot dans chaque classe
    O_class_0 = np.sum(class_0_docs[:, word_index])
    O_class_1 = np.sum(class_1_docs[:, word_index])

    # Total d'occurrences du mot dans toutes les classes
    total_word_occurrences = O_class_0 + O_class_1

    # Nombre total de documents dans chaque classe
    total_class_0_docs = class_0_docs.shape[0]
    total_class_1_docs = class_1_docs.shape[0]
    total_docs = total_class_0_docs + total_class_1_docs

    # Fréquence attendue du mot dans chaque classe avec pondération
    E_class_0 = (total_word_occurrences * (total_class_0_docs / total_docs)) * class_weights[0]
    E_class_1 = (total_word_occurrences * (total_class_1_docs / total_docs)) * class_weights[1]

    # Éviter la division par zéro
    if E_class_0 == 0 or E_class_1 == 0:
        return 0

    # Calcul du score Chi² avec pondération
    chi_square_class_0 = ((O_class_0 - E_class_0) ** 2) / E_class_0
    chi_square_class_1 = ((O_class_1 - E_class_1) ** 2) / E_class_1

    chi_square = chi_square_class_0 + chi_square_class_1
    return chi_square

def test_and_remove_words_weighted(data_train, labels_train, vocab_map, num_words=25, chi2_threshold=3.84):
    """
    Teste les N mots les plus fréquents selon le test Chi² pondéré et supprime ceux qui sont non significatifs.

    Arguments :
    - data_train : numpy array ou sparse matrix, matrice des termes (documents x termes)
    - labels_train : pandas DataFrame, étiquettes de classe pour chaque document
    - vocab_map : list, vocabulaire (termes associés aux colonnes de data_train)
    - num_words : int, nombre de mots à tester (par défaut 25)
    - chi2_threshold : float, seuil de significativité pour le test Chi² (par défaut 3.84 pour alpha = 0.05)

    Retourne :
    - data_train_filtered : sparse matrix ou numpy array, matrice des termes nettoyée
    - vocab_map_filtered : list, vocabulaire nettoyé
    """
    # Calculer la fréquence des termes (somme des occurrences par colonne)
    term_frequencies = np.array(np.sum(data_train, axis=0)).flatten()

    # Utiliser np.argpartition pour trouver les num_words mots les plus fréquents plus efficacement
    top_word_indices = np.argpartition(term_frequencies, -num_words)[-num_words:]

    # Calculer les poids des classes en fonction de la taille des classes
    class_weights = {
        0: len(labels_train[labels_train['label'] == 1]) / len(labels_train),
        1: len(labels_train[labels_train['label'] == 0]) / len(labels_train)
    }

    # Parcourir chaque mot parmi les plus fréquents
    for idx in sorted(top_word_indices, reverse=True):  # Inverser pour éviter de supprimer dans le mauvais ordre
        chi_square_score = chi_square_test_for_word_with_weights(idx, data_train, labels_train, class_weights)
        
        # Afficher le progrès et les résultats du test
        print(f"Terme {vocab_map[idx]}, Score Chi² : {chi_square_score:.4f}")
        
        # Vérifier si le score Chi² est inférieur au seuil, et supprimer les mots non significatifs
        if chi_square_score < chi2_threshold:
            print(f"Terme {vocab_map[idx]} supprimé (non significatif)")
            data_train = np.delete(data_train, idx, axis=1)
            vocab_map.pop(idx)
        else:
            print(f"Terme {vocab_map[idx]} conservé (significatif)")

    return data_train, vocab_map



def low_frequency_words(X, vocab_map, num_words=25):
    """
    Retourne les mots avec les fréquences les plus faibles dans une matrice de comptage de termes,
    ainsi que leurs fréquences.

    Arguments :
    - X : numpy array, matrice des termes (documents x termes)
    - vocab_map : list, vocabulaire (termes associés aux colonnes de X)
    - num_words : int, nombre de mots à retourner (par défaut 25)

    Retourne :
    - low_freq_words : list, les mots avec les fréquences les plus faibles
    - low_freq_values : list, les fréquences correspondantes des mots
    """
    # Calculer la fréquence totale de chaque mot dans l'ensemble des documents
    term_frequencies = np.sum(X, axis=0)
    
    # Obtenir les indices des mots avec les fréquences les plus faibles
    low_freq_indices = np.argsort(term_frequencies)[:num_words]
    
    # Récupérer les mots et leurs fréquences correspondants
    low_freq_words = [vocab_map[idx] for idx in low_freq_indices]
    low_freq_values = [term_frequencies[idx] for idx in low_freq_indices]
    
    return low_freq_words, low_freq_values


def find_distinctive_terms(data_train, labels_train, vocab_map, top_n=10):
    """
    Trouve les termes les plus distinctifs pour chaque classe en utilisant les fréquences normalisées.
    
    Arguments:
    - data_train : numpy array, matrice terme-document (chaque ligne est un document, chaque colonne est un mot)
    - labels_train : numpy array, labels de classe pour chaque document
    - vocab_map : liste des mots du vocabulaire (chaque index correspond à un mot)
    - top_n : int, nombre de termes les plus distinctifs à retourner pour chaque classe
    
    Retourne :
    - class_0_terms : DataFrame des termes les plus distinctifs pour la classe 0
    - class_1_terms : DataFrame des termes les plus distinctifs pour la classe 1
    """
    # Séparer les documents par classe
    class_0_docs = data_train[labels_train == 0]
    class_1_docs = data_train[labels_train == 1]
    
    # Calculer les fréquences des mots pour chaque classe
    class_0_word_frequencies = np.sum(class_0_docs, axis=0) + 1  # Lissage additif (Laplace)
    class_1_word_frequencies = np.sum(class_1_docs, axis=0) + 1  # Lissage additif (Laplace)
    
    # Normaliser les fréquences
    class_0_word_probabilities = class_0_word_frequencies / np.sum(class_0_word_frequencies)
    class_1_word_probabilities = class_1_word_frequencies / np.sum(class_1_word_frequencies)
    
    # Calculer la différence des probabilités normalisées
    distinctive_scores = class_1_word_probabilities - class_0_word_probabilities
    
    # Trouver les termes les plus distinctifs pour chaque classe
    class_0_distinctive_indices = np.argsort(distinctive_scores)[:top_n]  # Top termes pour la classe 0
    class_1_distinctive_indices = np.argsort(distinctive_scores)[-top_n:]  # Top termes pour la classe 1

    class_0_terms = [(vocab_map[idx], distinctive_scores[idx]) for idx in class_0_distinctive_indices]
    class_1_terms = [(vocab_map[idx], distinctive_scores[idx]) for idx in class_1_distinctive_indices]

    # Convertir en DataFrame pour une meilleure lisibilité
    class_0_terms_df = pd.DataFrame(class_0_terms, columns=["Term", "Score"]).sort_values(by="Score", ascending=True)
    class_1_terms_df = pd.DataFrame(class_1_terms, columns=["Term", "Score"]).sort_values(by="Score", ascending=False)
    
  
    return class_0_terms_df, class_1_terms_df

# Fonction pour afficher les termes les plus fréquents par classe
def print_top_terms_per_class(data, labels, vocab, top_n=25):
    # Extraire uniquement la colonne 'label' pour obtenir les étiquettes sous forme de vecteur
    
    classes = np.unique(labels)
    for cls in classes:
        # Sélectionner les documents de la classe actuelle en utilisant les labels
        class_data = data[labels == cls]
        
        # Calculer les fréquences des termes pour cette classe
        term_frequencies = np.sum(class_data, axis=0)
        
        # Identifier les indices des termes les plus fréquents pour la classe
        top_indices = np.argsort(term_frequencies)[-top_n:]
        
        # Afficher les termes les plus fréquents pour cette classe
        print(f"\nClasse {cls} :")
        for idx in reversed(top_indices):  # Utilisation de reversed pour trier en ordre décroissant
            print(f"Terme : {vocab[idx]}, Fréquence : {term_frequencies[idx]}")


def class_specific_words(data_train, labels_train, vocab_map):
    """
    Analyse les mots qui apparaissent exclusivement dans une seule classe, retourne les mots triés 
    par fréquence décroissante et trace la distribution.

    Arguments:
    - data_train : numpy array, matrice terme-document (chaque ligne est un document, chaque colonne est un mot)
    - labels_train : numpy array, labels de classe pour chaque document
    - vocab_map : liste des mots du vocabulaire (chaque index correspond à un mot)

    Retourne :
    - mots_classe_0 : liste des tuples (mot, fréquence) pour la classe 0 triée par fréquence décroissante
    - mots_classe_1 : liste des tuples (mot, fréquence) pour la classe 1 triée par fréquence décroissante
    """
    # Séparer les documents par classe
    data_class_0 = data_train[labels_train == 0]
    data_class_1 = data_train[labels_train == 1]

    # Calculer le nombre de documents dans lesquels chaque mot apparaît par classe
    word_counts_class_0 = np.sum(data_class_0 > 0, axis=0)
    word_counts_class_1 = np.sum(data_class_1 > 0, axis=0)

    # Identifier les mots qui apparaissent exclusivement dans une seule classe
    mots_classe_0_indices = np.where((word_counts_class_0 > 0) & (word_counts_class_1 == 0))[0]
    mots_classe_1_indices = np.where((word_counts_class_1 > 0) & (word_counts_class_0 == 0))[0]

    # Créer des listes triées par fréquence décroissante
    mots_classe_0 = [(vocab_map[idx], word_counts_class_0[idx]) for idx in mots_classe_0_indices]
    mots_classe_0 = sorted(mots_classe_0, key=lambda x: x[1], reverse=True)

    mots_classe_1 = [(vocab_map[idx], word_counts_class_1[idx]) for idx in mots_classe_1_indices]
    mots_classe_1 = sorted(mots_classe_1, key=lambda x: x[1], reverse=True)

    # Distribution pour la classe 0
    frequence_classe_0 = [freq for _, freq in mots_classe_0]
    plt.figure(figsize=(10, 5))
    plt.hist(frequence_classe_0, bins=20, color='blue', alpha=0.7, label='Classe 0')
    plt.title("Distribution des mots spécifiques à la classe 0")
    plt.xlabel("Nombre de documents")
    plt.ylabel("Nombre de mots")
    plt.legend()
    plt.show()

    # Distribution pour la classe 1
    frequence_classe_1 = [freq for _, freq in mots_classe_1]
    plt.figure(figsize=(10, 5))
    plt.hist(frequence_classe_1, bins=20, color='red', alpha=0.7, label='Classe 1')
    plt.title("Distribution des mots spécifiques à la classe 1")
    plt.xlabel("Nombre de documents")
    plt.ylabel("Nombre de mots")
    plt.legend()
    plt.show()

    print("Nombre de mots spécifiques à la classe 0 :", len(mots_classe_0))
    print("Nombre de mots spécifiques à la classe 1 :", len(mots_classe_1))

    return mots_classe_0, mots_classe_1




def extract_years_from_class(data_train, labels_train, vocab_map):
    """
    Extrait les années (4 chiffres consécutifs) pour chaque classe de documents et calcule leur fréquence.
    
    Arguments:
    - data_train : numpy array, matrice terme-document
    - labels_train : numpy array, labels de classe pour chaque document
    - vocab_map : liste ou numpy array des mots du vocabulaire
    
    Retourne :
    - years_by_class : dictionnaire où chaque clé est une classe et chaque valeur est un Counter des années et leurs fréquences.
    """
    # Convertir vocab_map en liste si ce n'est pas déjà le cas
    vocab_list = list(vocab_map)

    # Identifier les mots qui correspondent à des années (ex. : 4 chiffres consécutifs)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    year_indices = [idx for idx, word in enumerate(vocab_list) if year_pattern.fullmatch(word)]

    # Séparer les données par classe
    data_class_0 = data_train[labels_train == 0]
    data_class_1 = data_train[labels_train == 1]

    # Compter les fréquences des années pour chaque classe
    years_class_0 = Counter()
    years_class_1 = Counter()

    for idx in year_indices:
        word = vocab_list[idx]
        years_class_0[word] += np.sum(data_class_0[:, idx])
        years_class_1[word] += np.sum(data_class_1[:, idx])

    # Retourner les fréquences pour chaque classe
    return {0: years_class_0, 1: years_class_1}



def visualisation(data_train, labels_train, vocab_map,log=False): 
    distribution_mot(data_train, labels_train, vocab_map,log)
    loi_de_Zipf(data_train, labels_train, vocab_map)
    nuage_termes_frequent(data_train, labels_train, vocab_map)
    distribution_classes(data_train, labels_train, vocab_map)
    distribution_document_par_classe(data_train, labels_train, vocab_map)
    distribution_cumule_frequence_termes(data_train, labels_train, vocab_map)
    distribution_longueur_document(data_train, labels_train, vocab_map)
    identificatino_termes_different(data_train, labels_train, vocab_map)
    identification_termes_differents_norm(data_train, labels_train, vocab_map)
    nuage_classe0(data_train, labels_train, vocab_map)
    nuage_classe1(data_train, labels_train, vocab_map)
    

    


