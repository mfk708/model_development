# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:27:43 2020

@author: moham
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()
from sklearn.linear_model import LinearRegression
lm = LinearRegression() #create the linear regression object.
lm
# =============================================================================
#predict price based on highway-mpg using linear regression:
X = df[['highway-mpg']]
Y = df[['price']]
lm.fit(X,Y) #Fit the linear model using highway-mpg.
Yhat=lm.predict(X) 
Yhat[0:5] 
lm.intercept_ #value of the intercept
lm.coef_ #value of the coefficient
#multiple linear regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z.dtypes
lm.fit(Z, df['price'])
lm.intercept_
lm.coef_

df[['peak-rpm', 'highway-mpg', 'price']].corr() #correlation 
#regression plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

df[['peak-rpm', 'highway-mpg', 'price']].corr() #correlations
#residual plot
#If the points in a residual plot are randomly spread out around the x-axis, then a linear model
#is appropriate for the data. Why is that? Randomly spread out residuals means that 
#the variance is constant, and thus the linear model is a good fit for this data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
#plot the distribution of fitted values that results from the model and compare it to the actual values. 
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
#polynomial regression
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
#create a PolynomialFeatures object of degree 2
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape
#Data Pipelines simplify the steps of processing the data. We use the module Pipeline to 
#create a pipeline. We also use StandardScaler as a step in our pipeline.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#create the pipeline, by creating a list of tuples including the name of the model
#or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
#input the list as an argument to the pipeline constructor
pipe=Pipeline(Input)
pipe
#normalize the data, perform a transform and fit the model simultaneously.
pipe.fit(Z,y)
#normalize the data, perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(Z)
ypipe[0:4]

#mean squared error
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
#predict "yhat" using the predict method, where X is the input variable
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
from sklearn.metrics import mean_squared_error
#compare the predicted results with the actual result
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
#calculate the R^2
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
    
#polynomial fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))

 