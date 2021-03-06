import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# read the data
dataset=pd.read_csv('boston.csv')

# output the data
print(dataset)

# Define features, target values
my_features=dataset[['crim','rm','lstat']]
my_targets=dataset[['medv']]

# output descriptive statistics of features.
my_features.info()
crim_avg=str(my_features['crim'].mean())
rm_avg=str(my_features['rm'].mean())
lstat_avg=str(my_features['lstat'].mean())
print('Features average:')
print('crim: '+crim_avg+'   rm: '+rm_avg+'   lstat: '+lstat_avg)

# split the data into training and test sets
x_train,x_test,y_train,y_test=train_test_split(my_features,my_targets,test_size=0.3)

# perform Linear Regression on the training set
lm=LinearRegression()
lm=lm.fit(x_train,y_train)

# output the regression equation coefficients and the intercept of the equation of the model
print("coefficients"+str(lm.coef_))
print("intercept"+str(lm.intercept_))

# Predict the test data set and output the predicted value
prediction=lm.predict(x_test)
plt.subplot(212)
plt.plot(prediction,color='c',label='predicted')
plt.legend()
plt.show()

# Calculate the MAE of the predicted and true values
print("MAE: "+str(metrics.mean_absolute_error(y_test,prediction)))

# Calculate the MSE of the predicted and true values
print("MSE: "+str(metrics.mean_squared_error(y_test,prediction)))


