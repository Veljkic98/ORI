from __future__ import print_function

import pandas as pd

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans




def load_credit_cards():
    rows = int(input("Koliko linija zelite da ucitate >> "))

    df = pd.read_csv('kartice1000prvih.csv', nrows=rows)
    data = []

    for x, y in zip(df['PURCHASES_FREQUENCY'], df['ONEOFF_PURCHASES']):
        data.append([x, y])

    return data

def load_excessive_spend(df, nrows):
    '''
    Ucitava kreditne kartice sa atributima limit i ukupan iznos potrosen na kupovoni

    :return:
    '''

    data = []

    # PURCHASES - ukupan iznos potrosen na kupovinu
    # CREDIT_LIMIT - limit kartice
    for x, y in zip(df['PURCHASES'], df['CREDIT_LIMIT']):
        print(str(x) + "   " + str(y))
        data.append([x, y])

    return data


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
    kmeans = KMeans(n_clusters=3, max_iter=100)
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

def plot_2_D(data, xlab, ylab):
    # iscrtavamo sve tacke
    plt.figure()
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

    # --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

    # ovo je prethodno bilo za Iris data set
    kmeans = KMeans(n_clusters=3, max_iter=100)
    kmeans.fit(data, normalize=False)

    colors = {0: 'red', 1: 'green', 2: 'blue'}  # , 3: 'purple'
    plt.figure()
    for idx, cluster in enumerate(kmeans.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
        for datum in cluster.data:  # iscrtavanje tacaka
            plt.scatter(datum[0], datum[1], c=colors[idx])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()



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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_3_D(data):
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
    p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
    p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

    host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)

    host.set_xlabel("Distance")
    host.set_ylabel("Density")
    par1.set_ylabel("Temperature")
    par2.set_ylabel("Velocity")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines])

    plt.show()


if __name__ == '__main__':
    nrows = int(input("Koliko linija zelite da ucitate >> "))

    df = pd.read_csv('kartice1000prvih.csv', nrows=nrows)

    # ovo ce biti za jednu kombinaciju. imacemo dosta vise
    data = load_excessive_spend(df, nrows)
    # optimal_k_plot(data)
    plot_2_D(data, 'Ukupno potroseno', 'Limit')


    # Isprobavam
    # plot_3_D(data)



