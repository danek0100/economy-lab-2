from src.task_1 import task_1
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd


def task_2(painter, stocks, level_VaR, df_for_graph, set_name, colour_base):
    data_frame_for_clustering = df_for_graph.copy()

    k_means = KMeans(n_clusters=10, random_state=0)
    data_frame_for_clustering['cluster'] = k_means.fit_predict(data_frame_for_clustering)

    centroids = k_means.cluster_centers_
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]

    # Добавление полученных кластеров в дата-фрейм?
    data_frame_for_clustering['cen_x'] = data_frame_for_clustering.cluster.map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2],
                                                                                3: cen_x[3], 4: cen_x[4], 5: cen_x[5],
                                                                                6: cen_x[6], 7: cen_x[7], 8: cen_x[8],
                                                                                9: cen_x[9]})

    data_frame_for_clustering['cen_y'] = data_frame_for_clustering.cluster.map({0: cen_y[0], 1: cen_y[1], 2: cen_y[2],
                                                                                3: cen_y[3], 4: cen_y[4], 5: cen_y[5],
                                                                                6: cen_y[6], 7: cen_y[7], 8: cen_y[8],
                                                                                9: cen_y[9]})
    # Опредление карты цветов.
    colors = ['#F5001D', '#FFAA00', '#FFEC40', '#87EA00', '#1B1BB3',
              '#33CCCC', '#1921B1', '#6C006C', '#E73A95', '#007536']

    data_frame_for_clustering['c'] = data_frame_for_clustering.cluster.map({0: colors[0], 1: colors[1], 2: colors[2],
                                                                            3: colors[3], 4: colors[4], 5: colors[5],
                                                                            6: colors[6], 7: colors[7], 8: colors[8],
                                                                            9: colors[9]})

    painter.plot_clusters(data_frame_for_clustering, 0.6, 10, 'Splitting stocks into clusters')
    painter.plot()

    Es = []
    risks = []

    best_ValAtRisks = [1.1 for _ in range(10)]
    selected_stocks = []
    for cluster_index in range(10):
        cluster_checked = False
        for stock_index in range(len(stocks)):
            if data_frame_for_clustering['cluster'][stock_index] == cluster_index and \
                    best_ValAtRisks[cluster_index] > stocks[stock_index].VaR[level_VaR]:
                if cluster_checked > 0:
                    selected_stocks.pop()
                    Es.pop()
                    risks.pop()
                best_ValAtRisks[cluster_index] = stocks[stock_index].VaR[level_VaR]
                selected_stocks.append(stocks[stock_index])
                Es.append(stocks[stock_index].E)
                risks.append(stocks[stock_index].risk)
                cluster_checked = True

    df_for_graph = pd.DataFrame(
        {'σ': risks,
         'E': Es
         })

    task_1(painter, selected_stocks, level_VaR, df_for_graph, set_name, colour_base)
