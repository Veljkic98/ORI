from __future__ import print_function

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from kmeans import KMeans



def load_data(df, best_cols):
    '''
    Ucitava kreditne kartice sa atributima limit i ukupan iznos potrosen na kupovoni

    :return:
    '''

    data = []

    for x1, x2, x3, x4, x5, x6 in zip(df[best_cols[0]], df[best_cols[1]], df[best_cols[2]], df[best_cols[3]],
                                      df[best_cols[4]], df[best_cols[5]]):
        if x6 != x6:
            x6 = 0
        data.append([x1, x2, x3, x4, x5, x6])

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
        Grupe za iscrtavanje redom:

            -Big Spenders with large Payments - they make expensive purchases and have a credit limit that is between average and high. This is only a small group of customers.
            -Cash Advances with large Payments - this group takes the most cash advances. They make large payments, but this appears to be a small group of customers.
            -Medium Spenders with third highest Payments - the second highest Purchases group (after the Big Spenders).
            -Highest Credit Limit but Frugal - this group doesn't make a lot of purchases. It looks like the 3rd largest group of customers.
            -Cash Advances with Small Payments - this group likes taking cash advances, but make only small payments.
            -Small Spenders and Low Credit Limit - they have the smallest Balances after the Smallest Spenders, their Credit Limit is in the bottom 3 groups, the second largest group of customers.
            -Smallest Spenders and Lowest Credit Limit - this is the group with the lowest credit limit but they don't appear to buy much. Unfortunately this appears to be the largest group of customers.
            -Highest Min Payments - this group has the highest minimum payments (which presumably refers to "Min Payment Due" on the monthly statement. This might be a reflection of the fact that they have the second lowest Credit Limit of the groups, so it looks like the bank has identified them as higher risk.)

    :param scores:
    :param xlab:
    :param ylab:
    :return:
    """

    k_means = do_k_means(scores, n_clusters=8, max_iter=100)

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'purple', 5: 'dimgrey', 6: 'olive', 7: 'cyan'}  # , 3: 'purple'
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


def main1():
    nrows = int(input("Koliko linija zelite da ucitate >> "))
    # nrows = 50
    '''
        Zasto bas ove kolone

        Ako budemo iscrtavali svih 18, imacemo mnogo ponavljanja.
        Kako ih ne bi imali uzecemo samo one koji se ne ponavljaju
    '''
    best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]

    df = pd.read_csv('credit_card_data.csv', nrows=nrows)
    df.dropna()  # izbaci red ili zameni sa nulom kolone koje su prazne

    data = load_data(df, best_cols)

    # pomocu sledeceg plota cemo videti koliko komponenti da uzmemo

    pca = PCA().fit(data)  # uzima sve komponente
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

    ###################

    pca = PCA(n_components=2)  # iz plota iznad gledamo oko 80%, i toliko setujemo broj komponenti
    pca.fit(data)
    scores = pca.transform(data)  # smanjujemo dimenzije na n_components
    print(scores)
    plot_2_D_2(scores, 'Komponenta 1', 'Komponenta 2')
    # pca_pd = pd.DataFrame(scores)
    # print(pca_pd)


def print_missing_values():
    """
    Ispisujemo kojih vrednosti i koliko nedostaje. Cisto reda radi.

    :return:
    """

    data = pd.read_csv('credit_card_data.csv')
    missing = data.isna().sum()
    print(missing)
    print("\n")


if __name__ == '__main__':
    print_missing_values()
    main1()




