from __future__ import print_function

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


#TODO: poseno iscrtati centre ispod, kako bi se videli na plotu
def plot_2_D(k_means):
    # iscrtavamo sve tacke
    colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple',
              5: 'brown', 6: 'pink', 7: 'indigo', 8: 'pink', 9: 'gray'}
    plt.figure()
    for idx, cluster in enumerate(k_means.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c='black', marker='x', s=100)
        for datum in cluster.data:
            plt.scatter(datum[0], datum[1], c=colors[idx])

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
    k_means.fit(scores)
    klaster_indeksi = k_means.klaster_indeksi
    print(klaster_indeksi)

    # lista_klastera_sa_originalnim_podacima = [[], [], [], [], [], []]  # lista klastera sa originalnim podacima

    lista_klastera_sa_originalnim_podacima = []  # lista klastera sa originalnim podacima
    for i in range(broj_klastera):
        lista_klastera_sa_originalnim_podacima.append([])

    for i in range(len(original_data)):
        lista_klastera_sa_originalnim_podacima[klaster_indeksi[i]].append(original_data[i])

    # printujem osobine i stablo odlucivanja
    print_descriptions(lista_klastera_sa_originalnim_podacima, columns)
    # pecin_print(lista_klastera_sa_originalnim_podacima, columns)
    # print_decision_tree(original_data, klaster_indeksi, columns)

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

            print()  # prazan red

        print('\n\n')


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


def prikaz2(finalDf, cl_centers):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('x osa', fontsize=15)
    ax.set_ylabel('y osa', fontsize=15)
    ax.set_title('Grupe korisnika', fontsize=20)

    clusters = [0, 1, 2]
    colors = ['r', 'g', 'b']
    markers = ['*', 'X', 'o']

    legend = ['Grupa 1', 'Grupa 2', 'Grupa 3',
              'Centar grupe 1', 'Centar grupe 2', 'Centar grupe 3']

    # for target, color in zip(clusters, colors):
    #     f = finalDf.loc[finalDf['CLUSTER'] == target]
    #     ax.scatter(f['x_axis'], f['y_axis'], c=color, s=50)
    #
    # for target, mark in zip(clusters, markers):
    #     ax.scatter(cl_centers[target, 0], cl_centers[target, 1], color='black', marker=mark, label='centroid')
    #
    # ax.legend(legend)
    # ax.grid()
    # plt.show()


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




