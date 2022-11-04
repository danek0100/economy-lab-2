df_for_graph = pd.DataFrame(
        {'σ': risks,
         'E': Es
         })

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

