# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:58:27 2023

@author: emili
"""

import pandas as pd # data science library o manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization library
import seaborn as sb #visualization library specific for data science, based on matplotlib 
energyfuel = pd.read_csv('Sweden consumption by fuel.csv') # loads a csv file into a dataframe
swedenrenewables = pd.read_csv('Sweden renewables share.csv')
swedenco2 = pd.read_csv('Sweden co2 emissions.csv')

#now i need to prepare two different sets of data for the two different regressions
#1st regression will be for obtaining the CO2 emissions
#2nd regression will be for obtaining the renewable share
#they have to be different because the data for renewable share it is only until 1990

#1st set of data for CO2 emissions
energyfuel = energyfuel.set_index('year')
swedenco2 = swedenco2.set_index('year')
co2merged = pd.merge(energyfuel, swedenco2, on='year', how='outer')
#the databases with fuels and with CO2 emission have different amount of data
#so we will clean from this data the NaN values wich are the years in which we only have one of both data
co2merged_clean = co2merged.dropna()
n_nan = co2merged_clean.isna().sum().sum()
print("Number of NaN values in co2merged:", n_nan) #check that there are cero NaN
co2merged_clean['Previous year emissions'] = co2merged_clean['Fossil CO2 emissions (Tons)'].shift(1)
co2merged_clean['Previous year emissions'].iloc[0] = 0

# Drop the first row of the DataFrame
co2merged_clean = co2merged_clean.drop(co2merged_clean.index[0])

#FEATURE SELECTION

# Define input and outputs
Z=co2merged_clean.values
Y=Z[:,8]
X=Z[:,[0,1,2,3,4,5,6,7,9]] 
#print(Y)
#print(X)

#Now i use the linear regression model to select the top 3 features as a ranking

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model=LinearRegression() # LinearRegression Model as Estimator
rfe1=RFE(model,n_features_to_select=1)# using 1 features
rfe2=RFE(model,n_features_to_select=2) # using 2 features
rfe3=RFE(model,n_features_to_select=3)# using 3 features
fit1=rfe1.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)
print( "Feature Ranking (Linear Model, 1 features): %s" % (fit1.ranking_)) #petroleum products
print( "Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_)) #petroleum, biomass
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit3.ranking_)) #petroleum, biomass, electricity

#kbest to try to guess what are the most relevant features 

from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

features=SelectKBest(k=8,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results) #
print(plt.show()) 

features=SelectKBest(k=8,score_func=f_regression) # uses f-test ANOVA
fit=features.fit(X,Y) ##calculates the scores using the score_function f_regression of the features
print(fit.scores_)
features_results=fit.transform(X) 

plt.bar([i for i in range(len(fit.scores_))], fit.scores_)

#this method shows that the most relevant feature is petroleum so we will use:
#the first three features from linear regressions and the most relevant following kbest

#LINEAR REGRESSION  i will use this method to find the 2023 consumption

from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm

#Create matrix from data frame
Z=co2merged_clean.values
#Identify output Y
Y=Z[:,8]
#Identify input Y USING THE FEATURES POWER -1, HOUR, HOLIDAY as it was the result
#suggested by feature selection
X=Z[:,[0,2,6]]
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#print(X_train)
#print(y_train)
from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plt.scatter(y_test,y_pred_LR)

#ERROR ANALISIS

MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MBE_LR=np.mean(y_test- y_pred_LR) #here we calculate MBE
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
NMBE_LR=MBE_LR/np.mean(y_test)
print(MAE_LR, MBE_LR,MSE_LR, RMSE_LR,cvRMSE_LR,NMBE_LR)

#SAVE MODEL

import pickle
with open('regrco2.pkl','wb') as file: pickle.dump(regr, file)

#now we want to create a regresion model for renewable share
#the process is similar
#2nd set of data for CO2 emissions
swedenren = swedenrenewables.set_index('year')
renmerged = pd.merge(energyfuel, swedenren, on='year', how='outer')
#the databases with fuels and with CO2 emission have different amount of data
#so we will clean from this data the NaN values wich are the years in which we only have one of both data
renmerged_clean = renmerged.dropna()
n_nan = renmerged_clean.isna().sum().sum()
print("Number of NaN values in co2merged:", n_nan) #check that there are cero NaN

#FEATURE SELECTION

# Define input and outputs
Z=renmerged_clean.values
Y=Z[:,8]
X=Z[:,[0,1,2,3,4,5,6,7]] 
#print(Y)
#print(X)

#Now i use the linear regression model to select the top 3 features as a ranking

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model=LinearRegression() # LinearRegression Model as Estimator
rfe1=RFE(model,n_features_to_select=1)# using 1 features
rfe2=RFE(model,n_features_to_select=2) # using 2 features
rfe3=RFE(model,n_features_to_select=3)# using 3 features
fit1=rfe1.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)
print( "Feature Ranking (Linear Model, 1 features): %s" % (fit1.ranking_)) #total (kwh)
print( "Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_)) #total kw(h), electricity
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit3.ranking_)) #petroleum, total (kwh), electricity

#kbest to try to guess what are the most relevant features 

from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

features=SelectKBest(k=8,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results) #
print(plt.show()) 

features=SelectKBest(k=8,score_func=f_regression) # uses f-test ANOVA
fit=features.fit(X,Y) ##calculates the scores using the score_function f_regression of the features
print(fit.scores_)
features_results=fit.transform(X) 

plt.bar([i for i in range(len(fit.scores_))], fit.scores_)

#this method shows that the most relevant feature is Total (kwh) so we will use:
#the first three features from linear regressions and the most relevant following kbest

#LINEAR REGRESSION  i will use this method to find the 2023 consumption

from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm

#LINEAR REGRESSION 

#Create matrix from data frame
Z=renmerged_clean.values
#Identify output Y
Y=Z[:,8]
#Identify input Y USING THE FEATURES POWER -1, HOUR, HOLIDAY as it was the result
#suggested by feature selection
X=Z[:,[2,6,7]]
X_train, X_test, y_train, y_test = train_test_split(X,Y)
print(X_train)
print(y_train)
from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plt.scatter(y_test,y_pred_LR)

#ERROR ANALISIS

MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MBE_LR=np.mean(y_test- y_pred_LR) #here we calculate MBE
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
NMBE_LR=MBE_LR/np.mean(y_test)
print(MAE_LR, MBE_LR,MSE_LR, RMSE_LR,cvRMSE_LR,NMBE_LR)

#the errors are inside the delimited expected values
#SAVE MODEL

import pickle
with open('regrren.pkl','wb') as file: pickle.dump(regr, file)




