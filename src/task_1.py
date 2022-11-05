import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from src.data import get_market_index
from src.functions import *


def task_1(stocks, level_VaR, df_for_graph):
    sns.set_style("darkgrid")
    sns.scatterplot(data=df_for_graph, x='σ', y='E', c='#6C8CD5', label='Stocks').set_title("Profitability/Risk Map")

    sigmas = []
    returns = []
    bounds = ((None, None),)
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)

    X_min_risk = optimize_portfolio(risk_function_for_portfolio_with_short_sales,
                                    r_matrix,
                                    mean_vec,
                                    cov_matrix,
                                    bounds)
    min_risk = risk_function_for_portfolio_with_short_sales(X_min_risk, cov_matrix)
    min_risk_return = np.dot(X_min_risk, mean_vec)

    target_range = np.linspace(min_risk_return, 0.05, 100)
    for portfolio_return in tqdm_notebook(target_range):
        X = optimize_portfolio(risk_function_for_portfolio_with_short_sales,
                               r_matrix,
                               mean_vec,
                               cov_matrix,
                               bounds,
                               target_return=portfolio_return)

        sigmas.append(risk_function_for_portfolio_with_short_sales(X, cov_matrix))
        returns.append(np.dot(X, mean_vec))

    plt.scatter(min_risk,
                   min_risk_return,
                   c='#BE008A',
                   marker='^',
                   s=300,
                   label='Portfolio with minimal risk. Short selling is allowed.',
                   edgecolors='white', )
    plt.plot(sigmas, returns, 'r--', label='Effective front. Short selling is allowed.')
    plt.legend(['Stock'])

    # Рассмотрим случай в котором короткие продажи запрещены
    sigmas_without_short_sales = []
    returns_without_short_sales = []
    bounds = ((0.0, 1.0),)
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)

    X_min_risk_without_short_sales = optimize_portfolio(risk_function_for_portfolio_with_short_sales,
                                                        r_matrix,
                                                        mean_vec,
                                                        cov_matrix,
                                                        bounds)

    min_risk_without_short_sales = risk_function_for_portfolio_with_short_sales(X_min_risk_without_short_sales,
                                                                                cov_matrix)
    min_risk_return_without_short_sales = np.dot(X_min_risk_without_short_sales, mean_vec)
    target_range = np.linspace(min_risk_return_without_short_sales, 0.1, 100)

    for portfolio_return in tqdm_notebook(target_range):
        X_without_short_sales = optimize_portfolio(risk_function_for_portfolio_with_short_sales,
                                                   r_matrix,
                                                   mean_vec,
                                                   cov_matrix,
                                                   bounds,
                                                   target_return=portfolio_return)

        sigmas_without_short_sales.append(
            risk_function_for_portfolio_with_short_sales(X_without_short_sales, cov_matrix))
        returns_without_short_sales.append(np.dot(X_without_short_sales, mean_vec))

    plt.plot(sigmas_without_short_sales, returns_without_short_sales, 'c--', label='Effective front. Short selling '
                                                                                      'prohibited.')
    plt.scatter(min_risk_without_short_sales,
                   min_risk_return_without_short_sales,
                   c='#3BDA00',
                   marker='^',
                   s=300,
                   edgecolors='white',
                   label='Portfolio with minimal risk. Short selling prohibited.')
    plt.legend(['Stock'])

    points = 100
    plt.plot(sigmas[:points], returns[:points], 'r--', label='Effective front. Short selling is allowed.')
    plt.plot(sigmas_without_short_sales, returns_without_short_sales, 'c--', label='Effective front. Short selling '
                                                                                      'prohibited.')

    plt.scatter(min_risk,
                   min_risk_return,
                   c='#BE008A',
                   marker='^',
                   s=300,
                   edgecolors='white',
                   label='Portfolio with minimal risk. Short selling is allowed.')

    plt.scatter(min_risk_without_short_sales,
                   min_risk_return_without_short_sales,
                   c='#3BDA00',
                   marker='^',
                   s=300,
                   edgecolors='white',
                   label='Portfolio with minimal risk. Short selling prohibited.')

    risk_equals = risk_function_for_portfolio_with_short_sales(np.ones(50) / 50, cov_matrix)
    return_equals = np.dot((np.ones(50) / 50), mean_vec)

    plt.scatter(risk_equals,
                return_equals,
                c='#72217D',
                marker='^',
                s=300,
                label='Balanced portfolio',
                edgecolors='white')

    index = get_market_index('^BVSP', level_VaR)

    plt.scatter(index.risk, index.E,
                   c='yellow',
                   marker='8',
                   s=300,
                   label='Market index - Bovespa',
                   edgecolors='white')
    plt.legend(['Stocks'])
    plt.legend()
