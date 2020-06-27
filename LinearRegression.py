
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm

dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size=0.2,random_state=1) 

#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#OLS Estimations (still not working on spyder)
X2=sm.add_constant(X_train)
model = sm.OLS(Y_train, X2)
result = model.fit()
print(result.summary())

#Predicting on test set
y_pred = regressor.predict(X_test)

#Visualizing the training set results
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') #plots the curve of a function 
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') #plots the curve of a function. arguments do not change because the reg line is made from a unique equation from training set. If we use X_test, y_pred. We get same line. 
plt.title('Salary Vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#finding intercerpt and coeff
print(regressor.coef_)
print(regressor.intercept_)

#finding prediction for 12 years of experience
print(regressor.predict([[12]])) #since the predict method always expects a 2D array as input