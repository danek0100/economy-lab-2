from data import *
from stock import Stock
from functions import *
import os 

import matplotlib.pyplot as plt

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
                          'GGBR4', 'GOLL4', 'HYPE3', 'ITSA4', 'ITUB4', 'JBSS3', 'RENT3', 'LREN3', 'MGLU3', 'MRFG3']

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
    sns.scatterplot(data=df_for_graph, x='σ', y='E').set_title("Profitability/Risk Map")
    plt.legend(["Stocks"])
  
plot_map(df_for_graph)
plt.show()

#===============================================================

def get_return_mean_cov(stocks): 
    
    # получить по выбранным активам матрицу их доходностей, 
    # вектор средних доходностей и матрицу ковариации
    r_matrix = {}
    for stock in stocks:
         r_matrix[stock.key] = stock.profitability
    r_df = pd.DataFrame(r_matrix).dropna()
    return r_df.values, r_df.mean().values, r_df.cov().values
