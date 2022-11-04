from data import *
from stock import Stock
from functions import *
import os 

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm_notebook
import tqdm


ids_path = "../resource/brazil_ids.csv"
level_VaR = '0.95'


# Провряем существует ли файл с id доступных акций.
if not os.path.isfile(ids_path):
    print("Ids didn't found")
    exit(1)

# Загружаем данные, если они не были загружены ранее.
if not os.path.isdir('../resource/data/'):
    get_data(ids_path)

# Формируем список акций из данных. Формируем логорифмические доходности, а также вычисляем оценки.
stocks = get_Stocks(level_VaR)


#Выбрали 50 рандомных акция, которые входят в состав индекса Bovespa
selected_stocks_names = [ 'QUAL3', 'RADL3', 'GOAU4', 'BEEF3', 'MRVE3', 'MULT3', 'NTCO3', 'PETR3', 'PETR4', 'PRIO3',  
                          'POSI3', 'ABEV3', 'LAME3', 'ARZZ3', 'B3SA3', 'BPAN4', 'BBSE3', 'BRML3', 'BBDC3', 'BBDC4',
                          'BRAP4', 'PCAR4', 'BRKM5', 'CCRO3', 'CIEL3', 'COGN3', 'CPLE6', 'CSAN3', 'CPFE3', 'CVCB3',
                          'CYRE3', 'DXCO3', 'ECOR3', 'ENBR3', 'EMBR3', 'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3', 'FLRY3',
                          'GGBR4', 'GOLL4', 'HYPE3', 'ITSA4', 'ITUB4', 'JBSS3', 'RENT3', 'LREN3', 'VALE3', 'MRFG3']

#print(len(selected_stocks_names))

selected_stocks = []
for name_stock in selected_stocks_names:
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks.append(find_stock)


Es = []
risks = []
for stock in selected_stocks:
    Es.append(stock.E)
    risks.append(stock.risk)


df_for_graph = pd.DataFrame(
        {'σ': risks,
         'E': Es
         })

def plot_map(df_for_graph):
    sns.set_style("darkgrid")
    sns.scatterplot(data=df_for_graph, x='σ', y='E', c = '#6C8CD5',label='Stocks').set_title("Profitability/Risk Map")
    plt.legend(["Stocks"])
  
plot_map(df_for_graph)
plt.show()

#===============================================================

def get_return_mean_cov(stocks): 
    # получить по выбранным активам матрицу их доходностей, 
    # вектор средних доходностей и матрицу ковариации
    r_matrix = {}
    min_data_len = len(min(stocks, key=lambda x: len(x.profitability[0])).profitability[0])
    for stock in stocks:
         r_matrix[stock.key] = stock.profitability[0].copy()[:min_data_len-1]
    print(r_matrix)
    r_df = pd.DataFrame(r_matrix).dropna()
    # print(r_df)
    return r_df.values, r_df.mean().values, r_df.cov().values

r_matrix, mean_vec, cov_matrix = get_return_mean_cov(selected_stocks)

# print(r_matrix)
# print(mean_vec)
# print(cov_matrix)


##Разрешено все:)
def risk_porfolio(X, cov_matrix, n_observations=1, sqrt=True):
    # риск портфеля
    if sqrt:
        return np.sqrt(np.dot(np.dot(X, cov_matrix), X.T))
    else:
        return np.dot(np.dot(X, cov_matrix), X.T) / np.sqrt(n_observations)

def optimize_portfolio(risk_porfolio,
                       returns,
                       mean_returns, 
                       cov_matrix, 
                       bounds,
                       target_return=None):
    
    # оптимизатор с итеративным методом МНК SLSQP
    # решает задачу мимнимизации уравнения Лагранжа 
    
    X = np.ones(returns.shape[1])
    X = X / X.sum()
    bounds = bounds * returns.shape[1]

    constraints=[]
    constraints.append({'type': 'eq', 'fun': lambda X: np.sum(X) - 1.0})
    if target_return:
        constraints.append({'type': 'eq', 
                            'args': (mean_returns,), 
                            'fun': lambda X, mean_returns: portfolio_return - np.dot(X, mean_returns)})

    return minimize(risk_porfolio, X,
                    args=(cov_matrix, returns.shape[0]), 
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds).x


psigmas = []
preturns = []
bounds = ((None, None),) 
r_matrix, mean_vec, cov_matrix = get_return_mean_cov(selected_stocks)

X_min_risk = optimize_portfolio(risk_porfolio,
                                r_matrix,
                                mean_vec,
                                cov_matrix,
                                bounds)
min_risk = risk_porfolio(X_min_risk, cov_matrix)#, sqrt=True)
min_risk_preturn = np.dot(X_min_risk, mean_vec)
target_range = np.linspace(min_risk_preturn, 0.05, 500)
    
for portfolio_return in tqdm_notebook(target_range):
    X = optimize_portfolio(risk_porfolio,
                           r_matrix,
                           mean_vec,
                           cov_matrix,
                           bounds, 
                           target_return=portfolio_return)
    psigmas.append(risk_porfolio(X, cov_matrix))#, sqrt=True))
    preturns.append(np.dot(X, mean_vec))


plot_map(df_for_graph)
plt.scatter(min_risk, 
            min_risk_preturn, 
            c='#BE008A',
            marker='^',
            s=300, 
            label='Portfolio with minimal risk. Short selling is allowed.', 
            edgecolors='white',)
plt.plot(psigmas, preturns, 'r--', label='Effective front. Short selling is allowed.')
plt.legend(['Stock'])
plt.legend()
plt.show()


#Запрещенно 

psigmas_ns = []
preturns_ns = []
bounds = ((0.0, 1.0),) 
r_matrix, mean_vec, cov_matrix = get_return_mean_cov(selected_stocks)


X_min_risk_ns = optimize_portfolio(risk_porfolio,
                                   r_matrix,
                                   mean_vec,
                                   cov_matrix,
                                   bounds)
min_risk_ns = risk_porfolio(X_min_risk_ns, cov_matrix)
min_risk_preturn_ns = np.dot(X_min_risk_ns, mean_vec)
target_range = np.linspace(min_risk_preturn_ns, 0.1, 500)


for portfolio_return in tqdm_notebook(target_range):
    X_ns = optimize_portfolio(risk_porfolio,
                                       r_matrix,
                                       mean_vec,
                                       cov_matrix,
                                       bounds, 
                                       target_return=portfolio_return)
    psigmas_ns.append(risk_porfolio(X_ns, cov_matrix))
    preturns_ns.append(np.dot(X_ns, mean_vec))

plot_map(df_for_graph)
plt.plot(psigmas_ns, preturns_ns, 'c--', label='Effective front. Short selling prohibited.')
plt.scatter(min_risk_ns, 
            min_risk_preturn_ns,
            c='#3BDA00',
            marker='^', 
            s=300, 
            edgecolors='white',
            label='Portfolio with minimal risk. Short selling prohibited.')
plt.legend(['Stock'])
plt.legend()
plt.show()



p = 100
plot_map(df_for_graph)
plt.plot(psigmas[:p], preturns[:p], 'r--', label='Effective front. Short selling is allowed.')
plt.plot(psigmas_ns, preturns_ns, 'c--', label='Effective front. Short selling prohibited.')

plt.scatter(min_risk,
            min_risk_preturn, 
            c='#BE008A',
            marker='^',
            s=300, 
            edgecolors='white',
            label='Portfolio with minimal risk. Short selling is allowed.')

plt.scatter(min_risk_ns, 
            min_risk_preturn_ns,
            c='#3BDA00',
            marker='^', 
            s=300, 
            edgecolors='white',
            label='Portfolio with minimal risk. Short selling prohibited.')

risk_pequals   = risk_porfolio(np.ones(50) / 50, cov_matrix ) 
return_pequals = np.dot((np.ones(50) / 50),  mean_vec)

plt.scatter(risk_pequals, 
            return_pequals,
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
plt.show()

