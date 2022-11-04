from data import *
from stock import Stock
from functions import *
import os 

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.cluster import KMeans


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


##Выбор посредством кластаризации - KMeans

dataframe_for_clustering = df_for_graph.copy()

kmeans = KMeans(n_clusters=10, random_state=0)
dataframe_for_clustering['cluster'] = kmeans.fit_predict(dataframe_for_clustering)

centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

## add to dataframe_for_clustering
dataframe_for_clustering['cen_x'] = dataframe_for_clustering.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3], 4:cen_x[4], 5:cen_x[5], 6:cen_x[6], 7:cen_x[7], 8:cen_x[8], 9:cen_x[9]})
dataframe_for_clustering['cen_y'] = dataframe_for_clustering.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3], 4:cen_y[4], 5:cen_y[5], 6:cen_y[6], 7:cen_y[7], 8:cen_y[8], 9:cen_y[9]})
# define and map colors
colors = ['#F5001D', '#FFAA00', '#FFEC40', '#87EA00', '#1B1BB3', '#33CCCC', '#1921B1', '#6C006C', '#E73A95', '#007536']
dataframe_for_clustering['c'] = dataframe_for_clustering.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3],  4:colors[4], 5:colors[5], 6:colors[6], 7:colors[7], 8:colors[8], 9:colors[9]})
plt.scatter(x = dataframe_for_clustering['σ'] , y = dataframe_for_clustering['E'], c = dataframe_for_clustering['c'],  alpha = 0.6, s=10)
plt.title('Splitting stocks into clusters')
plt.xlabel("σ")
plt.ylabel("E")
plt.show()

