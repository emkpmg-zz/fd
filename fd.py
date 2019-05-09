# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:36:52 2019

@author: PIANDT
"""
 
import numpy as np
from otherFunctions import fruitCluster
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

fruitsData = pd.read_table('fruit_data_with_colors.txt')

#summary of first 5 rows
fruitsData.head()

#summarized analyses of quantitative data
fruitsData.describe()

#1-feature engineering, choosing classifier
# create a dictionary of fruit type (fruit_label) which is numeric
# and map to the actual fruit name (fruit_name). 
#This gives a true representation of our output labels or anything we want to classify
individualFruitNames = dict(zip(fruitsData.fruit_label.unique(), fruitsData.fruit_name.unique()))   

#The fruitsData has information about fruit name (numeric and string), height, 
# mass, color score and width of selected fruits.
#heights -- how tall is the fruit 
#widths -- how wide is the fruit
#mass -- how heavy is the fruit

#Feature exploration and Analyses
#Scatter matrix checks whether numeric variables are correlated 
#and if correlation is positive or negative

X = fruitsData[['height', 'width', 'mass', 'color_score']]
y = fruitsData['fruit_label']

#spliting dataset into training and test sets for exploratory purpose 
XTrainSet, XTestSet, YTrainSet, YTestSet = train_test_split(X, y, random_state=0)

colorMap = cm.get_cmap('gnuplot')

# plotting a scatter matrix
scatterMatrix = scatter_matrix(XTrainSet, c= YTrainSet, marker = '*', s=40, hist_kwds={'bins':20}, figsize=(10,10), cmap=colorMap)

# 3D scatter plot
fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(XTrainSet['width'], XTrainSet['height'], XTrainSet['color_score'], c = YTrainSet, marker = '*', s=300)
ax.set_xlabel('Fruit Width')
ax.set_ylabel('Fruit Height')
ax.set_zlabel('color_score')
plt.show()

# After detecting the most prominent variables from our data visualization,
#we select  mass, width, and height as features for our classification
X = fruitsData[['mass', 'width', 'height']]
Y = fruitsData['fruit_label']

#spliting dataset into training and test sets for classification purpose after choosing relevant vars
#- 75/25 default if unspecified
XTrainSet, XTestSet, YTrainSet, YTestSet = train_test_split(X, Y, random_state=0)

#check sensitivity of KNN classifier accuracy to K size
knnRange = range(1,20)
knnAccuracy = []
for i in knnRange:
    knnClassifier = KNeighborsClassifier(n_neighbors = i)
    knnClassifier.fit(XTrainSet, YTrainSet)
    knnAccuracy.append(knnClassifier.score(XTestSet, YTestSet))
plt.figure()
plt.xlabel('K-Size')
plt.ylabel('Model accuracy')
plt.scatter(knnRange, knnAccuracy)
plt.xticks([0,5,10,15,20]);

#check sensitivity of KNN classifier accuracy to training and test split sizes
#using K size 3 
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors = 3)
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training set proportion (%)')
plt.ylabel('Model accuracy');


#choose a classifier -- with the best K size == 3 in this case
knnClassifier = KNeighborsClassifier(n_neighbors = 3)

# train classifer with training data
knnClassifier.fit(XTrainSet, YTrainSet)

#check mean accuracy on training data
modelAccuracy = knnClassifier.score(XTrainSet, YTrainSet)
print('Accuracy of KNN classifier :', modelAccuracy)

# Make predictions on unseen data. A small fruit with mass, width, height
fruitPredict = knnClassifier.predict([[80, 5.3, 7.5]])
print('Predicted fruit1: ', individualFruitNames[fruitPredict[0]])

fruitPredict = knnClassifier.predict([[180, 7.3, 8.5]])
print('Predicted fruit2: ', individualFruitNames[fruitPredict[0]])

#Decision Boundary plots with k=3 
fruitCluster(XTrainSet, YTrainSet, 3, 'uniform')













