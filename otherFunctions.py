import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import matplotlib.patches as mpatches



def fruitCluster(X, Y, n_neighbors, weights):
    if isinstance(X, (pd.DataFrame,)):
        matrixX = X[['height', 'width']].as_matrix()
        matrixY = Y.as_matrix()
    elif isinstance(X, (np.ndarray,)):
        # When X was scaled is already a matrix
        matrixX = matrixY[:, :2]
        matrixY = Y.as_matrix()
        print(matrixY)

    # Create color maps
    ColorMap = ListedColormap(['#ffb847', '#ff8b00', '#97d15f','#00ff00'])
    #boldColorMap  = ListedColormap(['#ffb847', '#ff8b00', '#97d15f','#00ff00'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(matrixX, matrixY)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
# mesh step size 
    meshStepSize = .01  
    plotSymbolSize = 50

    x_min, x_max = matrixX[:, 0].min() - 1, matrixX[:, 0].max() + 1
    y_min, y_max = matrixX[:, 1].min() - 1, matrixX[:, 1].max() + 1
    XX, YY = np.meshgrid(np.arange(x_min, x_max, meshStepSize),
                         np.arange(y_min, y_max, meshStepSize))

    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Plotting output in color
    Z = Z.reshape(XX.shape)
    plt.figure()
    plt.pcolormesh(XX, YY, Z, cmap=ColorMap)

    # Plot training points
    plt.scatter(matrixX[:, 0], matrixX[:, 1], s=plotSymbolSize, c=Y, cmap=ColorMap, edgecolor = 'black')
    plt.xlim(XX.min(), XX.max())
    plt.ylim(YY.min(), YY.max())

    Cluster0 = mpatches.Patch(color='#ffb847', label='apple')
    Cluster1 = mpatches.Patch(color='#ff8b00', label='mandarin')
    Cluster2 = mpatches.Patch(color='#97d15f', label='orange')
    Cluster3 = mpatches.Patch(color='#00ff00', label='lemon')
    plt.legend(handles=[Cluster0, Cluster1, Cluster2, Cluster3])


    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')

    plt.show()
