#dataset: https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style

#Names: Trevor Henderson, Aidan Gonzales

#import the needed libraries to feature engineer and k-means cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv('/home/trevor/Desktop/IS 170/student_performance.csv')

#datacleaning for kmeans. the data was pretty clean so I dropped tabels that I believe would compromise the algorithm and dropped empty cells
df1=df.copy()

df1= df1.dropna()

df2=df1.copy()

df2=df2.drop(columns=['StudyHours', 'Resources', 'Extracurricular', 'Motivation', 'Internet', 'Gender', 'Age', 'LearningStyle', 'OnlineCourses', 'Discussions', 'AssignmentCompletion', 'ExamScore', 'EduTech', 'StressLevel'])

print(df2.head(20))

#went with k=2 after testing different combinations to get a higher silhouette score
k=2
X=np.array(list(zip(df2['Attendance'], df2['FinalGrade'])))

#setting the cluster to k=2 and setting centroids
kmeans=KMeans(n_clusters=k)

#fit the model to data
kmeans=kmeans.fit(X)

#assign each student to a cluster
labels=kmeans.predict(X)

#get center point for each cluster
centroids=kmeans.cluster_centers_

#map labels to colors
c=['b','r','y','g','c','m']
colors=[c[i]for i in labels]

#plot students as colored dots according to their cluster
plt.scatter(df2['Attendance'], df2['FinalGrade'],c=colors, s=2)

#plot center point as black star like shown in the example
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=100, c='black')

#label the x and y axis
plt.xlabel('Attendance')
plt.ylabel('FinalGrade')

#print centroid coordinates
print(centroids)
score2= silhouette_score(X, labels)


#redoing using different k value to see if the score improves or decreases

k=3
X2=np.array(list(zip(df2['Attendance'], df2['FinalGrade'])))

kmeans=KMeans(n_clusters=k)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

c=['b','r','y','g','c','m']
colors=[c[i] for i in labels]

plt.scatter(df['Attendance'],df['FinalGrade'],c=colors,s=2)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=100,c='black')

print(centroids)
score3=silhouette_score(X2, labels)


#redoing for k=4 as well

k=4
X3=np.array(list(zip(df2['Attendance'], df2['FinalGrade'])))

kmeans=KMeans(n_clusters=k)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

c=['b','r','y','g','c','m']
colors=[c[i] for i in labels]

plt.scatter(df['Attendance'],df['FinalGrade'],c=colors,s=2)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=100,c='black')

print(centroids)
score4=silhouette_score(X3, labels)
#print the silhouette score (s>0.5 means a good fit)
print('Score for k=2 is ', score2)
print('Score for k=3 is ', score3)
print('Score for k=4 is ', score4)

#Summary
#we can see that as we increase the k value, the score decreases so we can assume that 2 clusters is a better fit for this data
#This divided students into 2 groups based on attendance and final grades
#One represents students with higher attendance and grades and the other represents the opposite
#This would be used in the real world to identify students who need extra support
#Schools could use this data as well to assign tutoring or attendance intervention to students underperforming (It was called saturday school at my high school)
