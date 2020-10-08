# Assignment 5: KNN
### Tina Jin and Virginia Weston

### Exploratory Data Analysis 
For our EDA, we identified the features of our model which were radius, texture, perimeter, area, smoothness, compactness, symmetry, and fractal dimension. Using these features and setting them to x, we were able to further explore relationships between the x variables and the outcome. Because the outcome (or the diagnosis result) in the dataset is a categorical variable M (malignant) or B (benign), we knew we would need to convert that to a numeral value. It was also evident that we would need to scale the data prior to running a KNN because of the differences in measurements between each feature. Furthermore, we ran a scatter plot matrix and heat map to visualize the given dataset. 
![](/images/Figure_2.png)
By using the heatmap we were able to conceptualize the linear relationships between the data. There were several notable positive linear relationships between the variables in the data. Although it may be obvious that the two features would be correlated, the strongest linear relationship in the data was between the area and the perimeter of a tumor with a correlation coefficient of 0.98. Another positive linear relationship with a correlation coefficient 0.68 is between a tumor’s symmetry and compactness. 
![](/images/Figure_1.png)

### Initial Manipulation/Feature Scaling
Since the outcome of malignant or benign tumors are labeled 'M' and 'B' instead of integers, we need to convert the labels to '1' and '0' first.
Then we applied different scalers to the features to see which one gives the best accuracy. We experimented Power Transformer, Standard Scaler, Max Absolute Scaler and Robust Scaler, and left out scalers like the Quantile Transformer that is known to distort spatial relationships. We decided that the Power Transformer Scaler was the best one. This scaler makes the features look more like Gaussian distribution.

### KNN
We looked over all hyperparameters in the scikitlearn KNeighborsClassifier API (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). Then we chose to adjust n_neighbors, weights, and metric parameters. The model produced the accuracy for 5, 10, 15, 20, 25, 30, 35-nearest neighbors as well as the execution time for each in order for us to find the parameter that produced the highest accuracy. The most important cap for KNN performance was the curse of dimensionality, but the execution time did not increase significantly as we increased the number of neighbors considered. We suspected that it was because of the overall sparsity of data, as there were only 100 data points. Changing the distance metric between 'euclidean' and 'manhattan' did not make too much difference, with the 'manhattan' metric being slightly better. Switching weights from 'uniform' to 'distance' did not make a huge difference in the accuracy either, but it significantly reduced execution time. 
![](/images/knn_k.png)

As a result of these tests of hyperparameters, we concluded that using 15 neighbors, a weight metric of distance (bias towards the nearest neighbors) and a distance metric of manhattan distance results in the best accuracy for the KNN model. We also calculated the precision of our KNN model by calling the sklearn.metrics.classification_report API. Below are the results that the model prints. The accuracy is 94%, and the precision is 93%.

![](/images/knn_metrics.png)


