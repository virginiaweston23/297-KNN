# Assignment 5: KNN
### Tina Jin and Virginia Weston

### Exploratory Data Analysis 
For our Exploratory Data Analysis, we initially ran a scatter plot matrix to explore relationships by setting the outcome (whether a person has diabetes or not) to the y variable and setting the rest to x. All of the x variables were quantitative, making it easier to decipher relationships between them.
![]()
Moreover, by using a heatmap, we were able to conceptualize any linear relationships between the data. The strongest linear relationship among the data was between age and pregnancy, with a correlation coefficient of 0.54. Although not considered a strong linear relationship, the two variables can be considered correlated to one another. 
![]()
### Initial Manipulation/Feature Scaling

### KNN
We looked over all hyperparameters in the scikitlearn KNeighborsClassifier API (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). Then we chose to adjust n_neighbors, weights, and metric parameters. The model produced the accuracy for 5, 10, 15, 20, 25, 30, 35-nearest neighbors as well as the execution time for each in order for us to find the parameter that produced the highest accuracy. The most important cap for KNN performance was the curse of dimensionality, but the execution time did not increase significantly as we increased the number of neighbors considered. We suspected that it was because of the overall sparsity of data, as there were only 100 data points. Changing the distance metric between 'euclidean' and 'manhattan' did not make too much difference, with the 'manhattan' metric being slightly better. Switching weights from 'uniform' to 'distance' did not make a huge difference in the accuracy either, but it significantly reduced execution time. As a result of these tests of hyperparameters, we concluded that using 30 neighbors, a weight metric of distance (bias towards the nearest neighbors) and a distance metric of manhattan distance results in the best accuracy for the KNN model.
![]()
We also calculated the precision of our KNN model by calling the sklearn.metrics.classification_report API. Below are the results that the model prints. The accuracy is 94%, and the precision is 93%.
![]()
![]()


