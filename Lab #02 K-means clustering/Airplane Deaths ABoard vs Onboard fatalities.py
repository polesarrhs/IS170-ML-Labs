#importing necessary libraries 
#dataset: https://www.kaggle.com/datasets/saurograndi/airplane-crashes-since-1908

#Names: Trevor Henderson, Aidan Gonzales

#importing pandas in order to organize data from CSV file
import pandas as pd
#importing numpy for numerical operations 
import numpy as np
#importing matplotlib for data visualization
import matplotlib.pyplot as plt
#importing scikit.cluster to perform k-means clustering on the dataset
from sklearn.cluster import KMeans

#feature engineering 

#reading CSV file "Airplane_Crashes_and_Fatalities_Since_1908.csv" and setting it as df
df=pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")
#creating a copy of the dataframe to work with
df1=df.copy()
#dropping unnecessary columns
df1.drop("Flight #",axis=1, inplace=True)
df1.drop("Operator",axis=1, inplace=True)
df1.drop("Time",axis=1, inplace=True)
df1.drop("Route",axis=1, inplace=True)
#dropping rows with missing values in critical columns
df1=df1.dropna(subset=["Date", "Location", "Fatalities", "Aboard","Ground"])
#renaming columns for better readability
df1.rename({'Ground': 'On Ground Fatalities'}, axis=1, inplace=True)
df1.rename({'Fatalities': 'OnBoard Fatalities'}, axis=1, inplace=True)
print(df1)

#Scatter plot to visualize the relationship between ABoard versus OnBoard Fatalities

#Scatter plot using Aboard and Onboard Fatalities 
plt.scatter(df1["Aboard"], df1["OnBoard Fatalities"])
#adding labels and title to the plot and showing graph 
plt.xlabel("Number of People Aboard")
plt.ylabel("Number of On OnBoard Fatalities")
plt.title("OnBoard Fatalities vs Number of People On Board")
plt.show()

#K-Means Clustering to identify patterns in the data

#Setting number of clusters to 2 for initial clustering
k=2
#creating a array with numpy to store Aboard and OnBoard Fatalities for clustering
X=np.array(list(zip(df1['Aboard'], df1['OnBoard Fatalities'])))
#setting model to KMeans with k clusters to fit model
model=KMeans(n_clusters=k)
#fitting the model to the data and predicting cluster labels
kmeans=model.fit(X)
labels=model.predict(X)
#getting the centroids of the clusters
centroids=model.cluster_centers_
print(labels)
print(centroids)

#Visualizing the clusters with different colors and centroids having an X shape

#setting c for colors of clusters and plotting points and centroids
c=['b','g']
#looping through each cluster to plot points and centroids
for i in range(k):
    points=np.array([X[j] for j in range(len(X)) if labels[j]==i])
    plt.scatter(points[:,0], points[:,1], c=c[i])
    plt.scatter(centroids[i,0], centroids[i,1], c='r', marker='x')      
#adding labels and title to the plot and showing graph
plt.xlabel("Number of People Aboard")
plt.ylabel("Number of OnBoard Fatalities")
plt.title("K-Means Clustering of OnBoard Fatalities")
plt.show()

#Predicting the cluster for a new data point with 150 people aboard and 100 on board fatalities
cluster=model.predict([[150, 100]])
print("The cluster for 150 people aboard and 100 on board fatalities is:", cluster)

#Evaluating the clustering performance using silhouette score for different clusters 

#importing silhouette_score from sklearn.metrics to evaluate clustering performance
from sklearn.metrics import silhouette_score
#calculating silhouette score for 2 clusters
score=silhouette_score(X, labels)
model.fit(X)
label=model.predict(X)
print("Silhouette Score:", score)

#calculating silhouette score for 3 clusters
models=KMeans(n_clusters=3)
models.fit(X)
labels=models.predict(X)
score=silhouette_score(X, labels)
print("Silhouette Score for 3 clusters:", score)

#calculating silhouette score for 4 clusters
models=KMeans(n_clusters=4)
models.fit(X)   
labels=models.predict(X)
score=silhouette_score(X, labels)
print("Silhouette Score for 4 clusters:", score)

#Visualizing the clusters for 4 clusters with different colors and centroids having an X shape

#setting k to 4 for clustering and c for colors of clusters
k=4
#c is set to different colors for each cluster
c=['b','g','y','m']
#looping through each cluster to plot points and centroids
for i in range(k):
    points=np.array([X[j] for j in range(len(X)) if labels[j]==i])
    plt.scatter(points[:,0], points[:,1], c=c[i])
    plt.scatter(models.cluster_centers_[i,0], models.cluster_centers_[i,1], c='r', marker='x')
#adding labels and title to the plot and showing graph
plt.xlabel("Number of People Aboard")
plt.ylabel("Number of OnBoard Fatalities")
plt.title("K-Means Clustering of OnBoard Fatalities with 4 Clusters")
plt.show()












