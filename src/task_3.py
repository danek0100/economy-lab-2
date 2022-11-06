from src.functions import *


def task_3(painter, stocks, df_for_graph):
    gammas = np.linspace(1, 5, 5)

    ####################################################################################################################
    # Короткие продажи разрешены.                                                                                      #
    ####################################################################################################################
    print("\nSituation when short sales is allowed\n")
    short_is_allowed = True

    risk_of_the_optimal_portfolio_with_minimal_risk, \
        profitability_of_the_optimal_portfolio_with_minimal_risk, \
        losses = risk_aversion_computing(stocks, short_is_allowed, gammas)

    message = "Portfolio. Short sales is allowed, confidence level = "
    painter.plot_map_with_portfolios(df_for_graph, risk_of_the_optimal_portfolio_with_minimal_risk,
                                     profitability_of_the_optimal_portfolio_with_minimal_risk, gammas, message,
                                     "Map for 1")
    painter.plot()
    VaR_for_portfolios(gammas, losses)

    ####################################################################################################################
    # Короткие продажи запрещены.                                                                                      #
    ####################################################################################################################
    print("\nSituation when short sales is prohibited\n")
    short_is_allowed = False
    risk_of_the_optimal_portfolio_with_minimal_risk_no_shorts, \
        profitability_of_the_optimal_portfolio_with_minimal_risk_no_shorts, \
        losses_not_short = risk_aversion_computing(stocks, short_is_allowed, gammas)

    message = "Portfolio. Short sales is prohibited, confidence level = "
    painter.plot_map_with_portfolios(df_for_graph, risk_of_the_optimal_portfolio_with_minimal_risk_no_shorts,
                                     profitability_of_the_optimal_portfolio_with_minimal_risk_no_shorts,
                                     gammas, message, "Map for 2")
    painter.plot()
    VaR_for_portfolios(gammas, losses_not_short)
