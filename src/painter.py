import seaborn as sns
import matplotlib.pyplot as plt


class Painter:
    def __init__(self):
        sns.set_style("darkgrid")
        self._number_of_figure = 1

    def plot_stock_map(self, df_for_graph, set_name, number_of_figure=0):
        if number_of_figure == 0:
            number_of_figure = self._number_of_figure
        plt.figure(number_of_figure)
        sns.scatterplot(data=df_for_graph, x='σ', y='E', c='#6C8CD5', label='Stocks').set_title(set_name)

    def plot_effective_point(self, min_risk, min_risk_return, color, label, number_of_figure=0):
        if number_of_figure == 0:
            number_of_figure = self._number_of_figure
        plt.figure(number_of_figure)
        plt.scatter(min_risk, min_risk_return, c=color, marker='^', s=300, label=label, edgecolors='white')
        plt.legend()

    def plot_effective_front(self, sigmas, returns, color, label, points=-1, number_of_figure=0):
        if number_of_figure == 0:
            number_of_figure = self._number_of_figure
        plt.figure(number_of_figure)
        plt.plot(sigmas[:points], returns[:points], color, label=label)
        plt.legend()

    def plot_point(self, risk, E, colour, marker, label, number_of_figure=0):
        if number_of_figure == 0:
            number_of_figure = self._number_of_figure
        plt.figure(number_of_figure)
        plt.scatter(risk, E, c=colour, marker=marker, s=300, label=label, edgecolors='white')
        plt.legend()

    def plot_clusters(self, data_frame_for_clustering, alpha, s, title, number_of_figure=0):
        if number_of_figure == 0:
            number_of_figure = self._number_of_figure
        plt.figure(number_of_figure)

        plt.scatter(x=data_frame_for_clustering['σ'], y=data_frame_for_clustering['E'],
                    c=data_frame_for_clustering['c'], alpha=alpha, s=s)
        plt.title(title)
        plt.xlabel("σ")
        plt.ylabel("E")
        plt.legend()

    def plot(self):
        self._number_of_figure += 1



