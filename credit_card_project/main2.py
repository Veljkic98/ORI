from __future__ import print_function

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kmeans import KMeans



def load_data(df):
    '''
    Ucitava kreditne kartice sa svih 17 atributa

    :return:
    '''

    data = df.iloc[:, 1:].values  # uzima sve kolone osim prve (ID)
    # data = df.values.toList()

    return data


def do_k_means(data, n_clusters, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    kmeans.fit(data, normalize=True)

    return kmeans


def plot_2_D(data, xlab, ylab, kmeans, x, y):
    # iscrtavamo sve tacke
    plt.figure()
    for i in range(len(data)):
        plt.scatter(data[i][x], data[i][y])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

    colors = {0: 'red', 1: 'green', 2: 'blue'}  # , 3: 'purple'
    plt.figure()
    for idx, cluster in enumerate(kmeans.clusters):
        plt.scatter(cluster.center[x], cluster.center[y], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
        for d in cluster.data:  # iscrtavanje tacaka
            plt.scatter(d[x], d[y], c=colors[idx])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def plot_2_D_2(scores, xlab, ylab):
    """

    :param scores:
    :param xlab:
    :param ylab:
    :return:
    """

    k_means = do_k_means(scores, n_clusters=4, max_iter=100)

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange'}#, 4: 'purple'}#, 5: 'dimgrey', 6: 'olive', 7: 'cyan'}  # , 3: 'purple'
    plt.figure()
    for idx, cluster in enumerate(k_means.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
        for d in cluster.data:  # iscrtavanje tacaka
            plt.scatter(d[0], d[1], c=colors[idx])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


# --- ODREDJIVANJE OPTIMALNOG K --- #
def optimal_k_plot(data):
    """
     Odredjivanje i iscrtavanje optimalnog K

    :param data:
    :return:
    """

    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(1, 10):
        k_means = KMeans(n_clusters=n_clusters, max_iter=300)
        k_means.fit(data)
        sse = k_means.sum_squared_error()
        sum_squared_errors.append(sse)

    plt.plot(sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('SSE')
    plt.show()


def components_plot(data):
    pca = PCA().fit(data)  # uzima sve komponente (17)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def main2():
    # nrows = int(input("Koliko linija zelite da ucitate >> "))

    # df = pd.read_csv('credit_card_data.csv', nrows=nrows)
    # df = pd.read_csv('credit_card_data.csv')
    df = pd.read_csv('credit_card_data.csv', nrows=1000)

    df.drop(columns=['CUST_ID'], inplace=True)
    df.dropna(inplace=True)
    df2 = df.copy(deep=True)  # kopija, trebace mi kasnije
    # print(df)
    df = df.fillna(df.median())
    # df = StandardScaler().fit_transform(df)  # skaliranje podataka
    # df.dropna()  # izbaci red ili zameni sa nulom kolone koje su prazne

    # print(df)
    # klasteri = 4, komponente = 5

    data = load_data(df)

    km = do_k_means(data=data, n_clusters=4, max_iter=100)

    lista_klastera = []


    # normalize_data(data)

    optimal_k_plot(data)  # gledamo optimalan broj klastera - elbow

    components_plot(data)  # gledamo koliko komponenti da uzmemo

    # good_number_of_clusters(data)

    ###################

    pca = PCA(n_components=5)  # iz plota iznad gledamo oko 80%, i toliko setujemo broj komponenti
    pca.fit(data)
    # scores = pca.fit_transform(df)
    scores = pca.transform(data)  # smanjujemo dimenzije na n_components
    print(scores)
    # scaledDf = pd.DataFrame(data=scores, columns=['x_axis', 'y_axis'])
    plot_2_D_2(scores, 'Komponenta 1', 'Komponenta 2')


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




