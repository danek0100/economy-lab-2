import pandas as pd
import matplotlib.pyplot as plt

from resource.constant_strings import *
from tqdm import tqdm_notebook
from src.data import get_market_index
from src.functions import *


def task_1(painter, stocks, level_VaR, df_for_graph, set_name, colour_base):
    painter.plot_stock_map(df_for_graph, set_name)
    painter.plot()
    sigmas = []
    returns = []
    bounds = ((None, None),)
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)

    X_min_risk = optimize_portfolio(risk_function_for_portfolio,
                                    r_matrix,
                                    mean_vec,
                                    cov_matrix,
                                    bounds)
    min_risk = risk_function_for_portfolio(X_min_risk, cov_matrix)
    min_risk_return = np.dot(X_min_risk, mean_vec)

    target_range = np.linspace(min_risk_return, 0.01, 100)
    for portfolio_return in tqdm_notebook(target_range):
        X = optimize_portfolio(risk_function_for_portfolio,
                               r_matrix,
                               mean_vec,
                               cov_matrix,
                               bounds,
                               target_return=portfolio_return)

        sigmas.append(risk_function_for_portfolio(X, cov_matrix))
        returns.append(np.dot(X, mean_vec))

    painter.plot_stock_map(df_for_graph, set_name)
    painter.plot_effective_point(min_risk, min_risk_return, '#BE008A',
                                 pwmr + ' ' + ssia)
    painter.plot_effective_front(sigmas, returns, 'r--', ef + ' ' + ssia)
    painter.plot()
    # Рассмотрим случай в котором короткие продажи запрещены
    sigmas_without_short_sales = []
    returns_without_short_sales = []
    bounds = ((0.0, 1.0),)
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)

    X_min_risk_without_short_sales = optimize_portfolio(risk_function_for_portfolio,
                                                        r_matrix,
                                                        mean_vec,
                                                        cov_matrix,
                                                        bounds)

    min_risk_without_short_sales = risk_function_for_portfolio(X_min_risk_without_short_sales, cov_matrix)
    min_risk_return_without_short_sales = np.dot(X_min_risk_without_short_sales, mean_vec)
    target_range = np.linspace(min_risk_return_without_short_sales, 0.1, 100)

    for portfolio_return in tqdm_notebook(target_range):
        X_without_short_sales = optimize_portfolio(risk_function_for_portfolio,
                                                   r_matrix,
                                                   mean_vec,
                                                   cov_matrix,
                                                   bounds,
                                                   target_return=portfolio_return)

        sigmas_without_short_sales.append(
            risk_function_for_portfolio(X_without_short_sales, cov_matrix))
        returns_without_short_sales.append(np.dot(X_without_short_sales, mean_vec))

    painter.plot_stock_map(df_for_graph, set_name)
    painter.plot_effective_point(min_risk_without_short_sales, min_risk_return_without_short_sales, '#3BDA00',
                                 pwmr + ' ' + ssp)
    painter.plot_effective_front(sigmas_without_short_sales, returns_without_short_sales, 'c--',
                                 ef + ' ' + ssp)
    painter.plot()

    points = 100
    painter.plot_stock_map(df_for_graph, set_name)
    painter.plot_effective_point(min_risk, min_risk_return, '#BE008A',
                                 pwmr + ' ' + ssia)
    painter.plot_effective_front(sigmas, returns, 'r--', ef + ' ' + ssia, points)
    painter.plot_effective_point(min_risk_without_short_sales, min_risk_return_without_short_sales, '#3BDA00',
                                 pwmr + ' ' + ssp)
    painter.plot_effective_front(sigmas_without_short_sales, returns_without_short_sales, 'c--',
                                 ef + ' ' + ssp)

    risk_equals = risk_function_for_portfolio(np.ones(len(stocks)) / len(stocks), cov_matrix)
    return_equals = np.dot((np.ones(len(stocks)) / len(stocks)), mean_vec)

    index = get_market_index('^BVSP', level_VaR)
    painter.plot_point(risk_equals, return_equals, '#72217D', '^', 'Balanced portfolio')
    painter.plot_point(index.risk, index.E, 'yellow', '8', 'Market index - Bovespa')
    painter.plot()

    # Сравнение
    colour_base *= 4
    base_colours = ['#BE008A', 'r--', '#3BDA00', 'c--', '#FFBD88', 'g--', '#FFCF48', 's--', '#422C15', 'b--', '#331414',
                    'y--']
    painter.plot_effective_point(min_risk, min_risk_return, base_colours[colour_base],
                                 pwmr + ' ' + ssia, 100)
    painter.plot_effective_front(sigmas, returns, base_colours[colour_base+1], ef + ' ' + ssia, points, 100)
    painter.plot_effective_point(min_risk_without_short_sales, min_risk_return_without_short_sales,
                                 base_colours[colour_base+2], pwmr + ' ' + ssp, 100)
    painter.plot_effective_front(sigmas_without_short_sales, returns_without_short_sales, base_colours[colour_base+3],
                                 ef + ' ' + ssp, -1, 100)

    if colour_base == 0:
        painter.plot_point(risk_equals, return_equals, '#72217D', '^', 'Balanced portfolio', 100)
        painter.plot_point(index.risk, index.E, 'yellow', '8', 'Market index - Bovespa', 100)
