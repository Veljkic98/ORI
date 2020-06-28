from __future__ import print_function

import pandas as pd

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans


def load_credit_cards():
    df = pd.read_csv('credit_card_data.csv')
    data = []

    for x, y in zip(df['PURCHASES_FREQUENCY'], df['ONEOFF_PURCHASES']):
        data.append([x, y])

    print(data)


# --- UCITAVANJE I PRIKAZ IRIS DATA SETA --- #
def iris():
    iris_data = load_iris()  # ucitavanje Iris data seta
    iris_data = iris_data.data[:, 1:3]  # uzima se druga i treca osobina iz data seta (sirina sepala i duzina petala)

    # iscrtavamo sve tacke
    plt.figure()
    for i in range(len(iris_data)):
        plt.scatter(iris_data[i, 0], iris_data[i, 1])

    plt.xlabel('Sepal width')
    plt.ylabel('Petal length')
    plt.show()


    # --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

    # TODO 2: K-means na Iris data setu
    kmeans = KMeans(n_clusters=4, max_iter=100)
    kmeans.fit(iris_data, normalize=True)

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
    plt.figure()
    for idx, cluster in enumerate(kmeans.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
        for datum in cluster.data:  # iscrtavanje tacaka
            plt.scatter(datum[0], datum[1], c=colors[idx])

    plt.xlabel('Sepal width')
    plt.ylabel('Petal length')
    plt.show()

    optimal_k_plot(iris_data)

# --- ODREDJIVANJE OPTIMALNOG K --- #
def optimal_k_plot(data):
    '''
     Odredjivanje i iscrtavanje optimalnog K

    :param data: Ucitani podaci
    :return:
    '''

    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
        kmeans.fit(data)
        sse = kmeans.sum_squared_error()
        sum_squared_errors.append(sse)

    plt.plot(sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('SSE')
    plt.show()


if __name__ == '__main__':
    iris()
    print("Pocetak")
    # load_credit_cards()



