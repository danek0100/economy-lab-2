import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm_notebook


# Печать карты (риск, доходность).
def plot_map(df_for_graph):
    fig = plt.figure()
    sns.set_style("darkgrid")
    sns.scatterplot(data=df_for_graph, x='σ', y='E', c='#6C8CD5', label='Stocks').set_title("Profitability/Risk Map")
    fig.legend(["Stocks"])
    return fig


def scatter_draw(stock1, stock2, keys):
    plt.grid(True)
    plt.scatter(stock1, stock2, edgecolors='white')
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.title(keys[2])
    plt.show()


def get_return_mean_cov(stocks):
    # получить по выбранным активам матрицу их доходностей,
    # вектор средних доходностей и матрицу ковариации
    r_matrix = {}
    min_data_len = len(min(stocks, key=lambda x: len(x.profitability[0])).profitability[0])
    for stock in stocks:
        r_matrix[stock.key] = stock.profitability[0].copy()[:min_data_len - 1]
    r_df = pd.DataFrame(r_matrix).dropna()
    return r_df.values, r_df.mean().values, r_df.cov().values


# Оценка портфелей для которыйх короткие продажи разрешены
def risk_function_for_portfolio(X, cov_matrix, n_observations=1, sqrt=True):
    # оценка риска портфеля
    if sqrt:
        return np.sqrt(np.dot(np.dot(X, cov_matrix), X.T))
    else:
        return np.dot(np.dot(X, cov_matrix), X.T) / np.sqrt(n_observations)


def optimize_portfolio(risk_estimation_function,
                       returns,
                       mean_returns,
                       cov_matrix,
                       bounds,
                       target_return=None):

    # оптимизатор с итеративным методом МНК SLSQP решает задачу мимнимизации уравнения Лагранжа
    X = np.ones(returns.shape[1])
    X = X / X.sum()
    bounds = bounds * returns.shape[1]

    constraints = [{'type': 'eq', 'fun': lambda X_: np.sum(X_) - 1.0}]
    if target_return:
        constraints.append({'type': 'eq',
                            'args': (mean_returns,),
                            'fun': lambda X_, mean_returns_: target_return - np.dot(X_, mean_returns_)})

    return minimize(risk_estimation_function, X,
                    args=(cov_matrix, returns.shape[0]),
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds).x
