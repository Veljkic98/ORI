from __future__ import print_function

from builtins import print

import numpy

import pandas as pd

import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import tree

from kmeans import KMeans as MyKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

import copy


def load_data(df):
    '''
    Ucitava kreditne kartice sa svih 17 atributa

    :return:
    '''

    data = df.iloc[:, 1:].values  # uzima sve kolone osim prve (ID)
    # data = df.values.toList()

    return data


def plot_2_D(k_means):
    # iscrtavamo sve tacke
    colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple',
              5: 'brown', 6: 'pink', 7: 'indigo', 8: 'pink', 9: 'gray'}
    plt.figure()
    for idx, cluster in enumerate(k_means.clusters):
        # plt.scatter(cluster.center[0], cluster.center[1], c='black', marker='x', s=100)
        for datum in cluster.data:
            plt.scatter(datum[0], datum[1], c=colors[idx])

    # iscrtavanje centara
    for idx, cluster in enumerate(k_means.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c='black', marker='x', s=100)

    plt.xlabel('X osa - komponenta')
    plt.ylabel('Y osa - komponenta')
    plt.show()


# --- ODREDJIVANJE OPTIMALNOG K --- #
def optimal_k_plot(data):
    """
     Odredjivanje i iscrtavanje optimalnog K

    :param data:
    :return:
    """

    sum_of_squared_distances = []

    K = range(1, 18)

    for k in K:
        km = MyKMeans(n_clusters=k, max_iter=150)
        km.fit(data)
        sse = km.sum_squared_error()
        sum_of_squared_distances.append(sse)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('# clusters')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def components_plot(data):
    pca = PCA().fit(data)  # uzima sve komponente (17)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def min_max_data(df, columns):
    print("*** Maksimalne i minimalne vrednosti za svaku kolonu ***\n\n")
    for c in columns:
        print(c)
        print("Max: " + str(df.iloc[:, 1:][c].max()))
        print("Min: " + str(df.iloc[:, 1:][c].min()))
    print("\n\n")


def main2():
    df = pd.read_csv('credit_card_data.csv')
    df = df.fillna(df.median())
    original_data = df.iloc[:, 1:].values
    data = copy.deepcopy(original_data)

    columns = list(df.columns)[1:]  # lista naziva kolona
    print(columns)

    # min_max_data(df, columns)

    normalizacija(data)  # radimo normalizaciju nad ucitanim podacima

    pca = PCA()
    pca.fit(data)

    # odredjujem na koliiko cu da smanjim dimenzionalnost
    plt.plot(range(1, 18), pca.explained_variance_ratio_.cumsum(), marker='x', linestyle='--')
    plt.xlabel('Components')  # features
    plt.ylabel('Variance')
    plt.show()

    components = 7  # vidimo iz plota
    pca = PCA(n_components=components)
    pca.fit(data)
    scores = pca.transform(data)
    # print(scores)  # ima onoliko komponenti koliko smo stavili

    # pokazujemo da prve dve dimenzije uticu najvise na grafik
    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA components')
    plt.ylabel('Variance %')  # procenat koliko uticu na grafik, da tako kazemo
    plt.xticks(range(pca.n_components_))
    plt.show()

    # dobijam optimal k = 5 za 500 prvih ucitanih
    # za sve ucitane dobijam 6
    # optimal_k_plot(data)

    broj_klastera = 6

    k_means = MyKMeans(n_clusters=broj_klastera, max_iter=100)
    k_means.fit(scores, normalize=False)
    klaster_indeksi = k_means.klaster_indeksi
    print(klaster_indeksi)

    lista_klastera_sa_originalnim_podacima = []  # lista klastera sa originalnim podacima
    for i in range(broj_klastera):
        lista_klastera_sa_originalnim_podacima.append([])

    for i in range(len(original_data)):
        lista_klastera_sa_originalnim_podacima[klaster_indeksi[i]].append(original_data[i])

    # printujem osobine i stablo odlucivanja
    print_descriptions(lista_klastera_sa_originalnim_podacima, columns)
    # print_decision_tree(original_data, klaster_indeksi, columns)
    print_clusters_description()

    # iscrtavamo tacke
    plot_2_D(k_means)


def print_descriptions(lista_klastera_sa_originalnim_podacima, columns):
    print("*** OPIS SVIH KLASTERA ***")
    for i, d in enumerate(lista_klastera_sa_originalnim_podacima, start=0):
        print('\nKlaster ' + str(i + 1))
        for j in range(len(columns)):
            print('Atribut: ' + columns[j] + ':')
            print('\tMinimum: ' + str(min([datum[j] for datum in d])))
            print('\tPrvi kvartil: ' + str(np.percentile([datum[j] for datum in d], 25)))
            print('\tMediana: ' + str(np.percentile([datum[j] for datum in d], 50)))
            print('\tTreci kvartil: ' + str(np.percentile([datum[j] for datum in d], 75)))
            print('\tMaksimum: ' + str(max([datum[j] for datum in d])))
            print('\tSrednja vrednost: ' + str(np.mean([datum[j] for datum in d])))
            print('\tUkupno korisnika: ' + str(len(d)))

            print()  # prazan red

        print('\n\n')


def print_clusters_description():
    print("*** Osobine klastera ***\n\n")

    print("Klaster 1:")
    print("\tU ovoj grupi korisnika, korisnici imaju vema malo stanje na racunu za kupovinu.\n"
          "\tTim je takodje i iznos za kupovinu veoma mali. Vise novca je potroseno na jednokratnu kupovinu\n"
          "\tnego na kupovinu na rate. Veoma mala suma uplacivanja novca unapred, za one koji uopste uplacuju.\n"
          "\tSkoro pola ih uopste neuplacuje unapred.\n\n")
    print("Klaster 2:")
    print("\tZa ovu grupu korisnicu u proseku trose cetvrtinu iznosa na kupovinu. Novac se skoro nikad ne uplacuje unapred.\n"
          "\tVecina kupuje jednokratno. Korisnici su raznovrsni sto se tice limita na kreditnoj kartici.\n"
          "\tVecina ima mali minimalni iznos uplacen na karticu.\n"
          "\tU ovoj grupi ima najvise korisnika.\n\n")
    print("Klaster 3:")
    print("\tNisko stanje na racunu dostupno za kupovinu. Malo se trosi na kupovinu.\n"
          "\tVise od pola korisnika ne placa jednokratno, dok na rate placaju svi, i prosek je veci za 4 puta.\n"
          "\tNovac se u ovoj grupi skoro nikad ne uplacuje unapred.\n"
          "\tOva grupa je po broju korisnika druga po velicini.\n\n")
    print("Klaster 4:")
    print("\tVeoma mali iznos potrosen na kupovinu. Jako retka jednokratna kupovina.\n\n")
    print("Klaster 5:")
    print("\tStanje na racunu dostupno za kupovinu je malo, a samim tim je i mali iznos potrosen na kupovinu.\n"
          "\tVise se kupovalo na rate nego jednokratno (jednokratno veoma retko).\n"
          "\tTakodje spadaju i korisnici kod kojih je iznos koji je uplacen unapred veoma mali i imaju mali limit\n"
          "\tna kreditnoj kartici. Ukupan iznos koji je uplacen na kraticu je mali.\n"
          "\tU ovoj grupi ima najmanje korisnika.\n\n")
    print("Klaster 6:")
    print("\tStanje na racunu dostupno za kupovinu je generalno malo, a samim tim i ukupan\n"
          "\tiznos potrosen na kupovinu mali. Potrosanja na rate je 4 puta veca od jendnokratne potrosnje,\n"
          "\tali je jednokratna potrosnja ucestalija. Iznos koji je korisnik uplatio unapred je u proseku veoma mali.\n\n")


def print_decision_tree(original_data, klaster_indeksi, columns):
    print("*** Stablo odlucivanja ***\n")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(original_data, klaster_indeksi)

    text_tree = tree.export_text(clf, feature_names=list(columns))
    print(text_tree)


def normalizacija(data):
    # mean-std normalizacija
    cols = len(data[0])

    for col in xrange(cols):
        column_data = []
        for row in data:
            column_data.append(row[col])

        mean = numpy.mean(column_data)
        std = numpy.std(column_data)

        for row in data:
            row[col] = (row[col] - mean) / std

    return data


def good_number_of_clusters(vals):
    wcss = []
    for ii in range(1, 30):
        kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
    kmeans.fit_predict(vals)
    wcss.append(kmeans.inertia_)

    plt.plot(wcss, 'ro-', label="WCSS")
    plt.title("Computing WCSS for KMeans++")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()


if __name__ == '__main__':
    main2()




