import pandas as pd
import matplotlib.pyplot as plt

from src.functions import *


def task_5(painter, stocks, df_for_graph):
    virtual_stock = count_virtual_stock_without_risk(stocks)
    painter.plot_stock_map(df_for_graph, "5 task")
    painter.plot_point(virtual_stock[0], virtual_stock[1], 'red', 'o', 'Virtual stock without risk')
    print("Risk stock ", virtual_stock[0])
    print("Return stock ", virtual_stock[1])

    ####################################################################################################################
    # Короткие продажи разрешены.                                                                                      #
    ####################################################################################################################
    short_is_allowed = True
    the_best_risk_sharp, the_best_E_sharp, losses = optimal_portfolio_computing(stocks, virtual_stock, short_is_allowed)
    message = "Optimal Sharpe portfolio. Short selling is allowed."
    color = '#A62500'
    painter.plot_portfolio(the_best_risk_sharp, the_best_E_sharp, message, color)
    print("Short selling is allowed.")
    VaR_for_portfolio(losses)

    ####################################################################################################################
    # Короткие продажи рзапешены.                                                                                      #
    ####################################################################################################################
    short_is_allowed = False
    the_best_risk_sharp_no_shorts, the_best_E_sharp_no_shorts, losses_no_shorts =\
        optimal_portfolio_computing(stocks, virtual_stock, short_is_allowed)
    message = "Optimal Sharpe portfolio. Short selling is prohibited."
    color = "#FF9273"
    painter.plot_portfolio(the_best_risk_sharp_no_shorts, the_best_E_sharp_no_shorts, message, color)
    painter.plot()
    print("Short selling is prohibited.")
    VaR_for_portfolio(losses_no_shorts)
