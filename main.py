#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)
from sklearn.ensemble import RandomForestRegressor


# read files
df_raw_energy = pd.read_csv('Energy_consumption_SE.csv', delimiter=';', header=0)
df_raw_population = pd.read_csv('Population_SE.csv', delimiter=';', header=0)

##Is sign for comment or explanation
#Is a code that is currently not active but can be activated if more graphics are needed

print(df_raw_energy.columns)
print(df_raw_population.columns)
##Change Year column to be the index
df_raw_energy = df_raw_energy.set_index('Year', drop=True)
df_raw_energy = df_raw_energy.rename(columns={'Total': 'Total Energy'})


##Population
df_raw_population = df_raw_population.set_index('Year')

##Combine all the data
df_clean = pd.merge(df_raw_energy, df_raw_population, how = 'outer', left_index=True, right_index=True)
df_clean = df_clean.drop('Unnamed: 2', axis=1)
df_clean.index = pd.to_datetime(df_clean.index.astype(str), format='%Y.0').strftime('%Y')
print(df_clean)

##If there is still missing values those are filled with zero.
##Just in case... Mostly because I can and otherwise there seemed to be a problem
df_clean = df_clean.dropna()


##Check that it worked
print("Missing values in a data set after cleaning: ")
print(df_clean.isnull().sum())
print(df_clean)
##Worked like a charm


##Check if there is outliers
#df_sort_industry = df_clean.sort_values(by = 'Total Energy', ascending = True)
df_sort_industry = df_clean.sort_values(by = 'Total Energy', ascending = False)
print("Total Energy")
print(df_sort_industry[:10])

df_sort_industry = df_clean.sort_values(by = 'Industry', ascending = True)
#df_sort_industry = df_clean.sort_values(by = 'Industry', ascending = False)
print("Industry outliers")
print(df_sort_industry[:10])

df_sort_transport = df_clean.sort_values(by = 'Domestic transport', ascending = True)
#df_sort_transport = df_clean.sort_values(by = 'Domestic transport', ascending = False)
print("Transport outliers")
print(df_sort_transport [:10])

df_sort_resident = df_clean.sort_values(by = 'Residential and services', ascending = True)
#df_sort_resident = df_clean.sort_values(by = 'Residential and services', ascending = False)
print("Residential and services outliers")
print(df_sort_resident [:10])

df_sort_population = df_clean.sort_values(by = 'Population', ascending = True)
#df_sort_population = df_clean.sort_values(by = 'Population', ascending = False)
print("Population outliers")
print(df_sort_population [:10])


##Check outliers by creating few plots of energy consumption data
print(df_clean[['Total Energy']].plot())
print(plt.show())

##Reorganise the columns
df_clean=df_clean.iloc[:, [3,0,1,2,4]]


##Based on the methods described above, there are no points where the energy consumption data is 0.

##Adding the energy consumption from previous years as separate columns
df_clean['Total-1'] = df_clean['Total Energy'].shift(1)
df_clean['Total-2'] = df_clean['Total Energy'].shift(2)

print(df_clean.columns)
##Remove rows with NaN from the begining
df_clean = df_clean.dropna()
##Checking the results
#print(df_clean[['Total Energy']].plot())
#print(plt.show())


##Feature selection

Z=df_clean.values
Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6]]

model=LinearRegression() # LinearRegression Model as Estimator
rfe1=RFE(model,n_features_to_select=1)# using 1 features
rfe2=RFE(model,n_features_to_select=2) # using 2 features
rfe3=RFE(model,n_features_to_select=3)# using 3 features

fit1=rfe1.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)

print( 'Feature Ranking (Linear Model, 1 features): %s' % (fit1.ranking_)) #Most important is industry
print( 'Feature Ranking (Linear Model, 2 features): %s' % (fit2.ranking_)) #Industry and Domestic transport
print( 'Feature Ranking (Linear Model, 3 features): %s' % (fit3.ranking_)) #Industry, Domestic transport and residential&services

##Feature selection with Kbest Method
##Another feature selection method
features=SelectKBest(k=3,score_func=f_regression) # uses f-test ANOVA
fit=features.fit(X,Y) ##calculates the scores using the score_function f_regression of the features
print(fit.scores_)    ##Industry, total energy-1 and Residential&services

##Another trial
features=SelectKBest(k=3,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) ##calculates the f_regression of the features
print(fit.scores_)    ##Industry, total energy-1 but others were very even


##According to these methods, Industry is the most important feature.
##Followed by energy-1 and Domestic transport

##Based on these results, the features used in this model are the energy consumption of the industry,
##total energy consumption from the previous year and domestic transport

##Next the useless columns are removed

#Save the old data frame for random forest
df_random = df_clean


##Reorganising and removing the columns
df_clean_dropped=df_clean.iloc[:, [0,5,1,2]]
print('This model uses following features')
columns_to_print = list(df_clean_dropped.iloc[:,[1,2,3]].columns)
print(columns_to_print)

##Creating a new file of the data cleaned
df_clean_dropped.to_csv('Final_df.csv', encoding='utf-8', index=True)

##training and testing data
Z=df_clean_dropped.values
Y=Z[:,0]
X=Z[:,[1,2,3]]

X_train, X_test, y_train, y_test = train_test_split(X,Y)

##Building a regression model by first testing three different models

##Decission tree method:
## Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()

##Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

## Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)

print(plt.plot(y_test[100:500]))
print(plt.plot(y_pred_DT[100:500]))
plt.title("Figure of Decission Tree")
print(plt.show())
print(plt.scatter(y_test,y_pred_DT))
plt.title("Title of the Scatter Plot")
print(plt.show())

MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT)
MBE_DT=np.mean(y_test-y_pred_DT) #here we calculate MBE
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
NMBE_DT=MBE_DT/np.mean(y_test)
print('Results from Decision tree')
print(MAE_DT, MBE_DT,MSE_DT, RMSE_DT,cvRMSE_DT,NMBE_DT)


##Neural network
NN_model = MLPRegressor(hidden_layer_sizes=(6,12,6), max_iter=250)
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)

plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.title("Figure of Neural Network")
print(plt.show())
print(plt.scatter(y_test,y_pred_NN))
plt.title("Title of the NN Scatter Plot")
print(plt.show())


MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN)
MBE_NN=np.mean(y_test-y_pred_NN)
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
NMBE_NN=MBE_NN/np.mean(y_test)
print('Neural network')
print(MAE_NN,MBE_NN,MSE_NN,RMSE_NN,cvRMSE_NN,NMBE_NN)

#import pickle
#with open('NN_model.pkl','wb') as file:
    #pickle.dump(NN_model, file)

##Random forest


X_train, X_test, y_train, y_test = train_test_split(X,Y)

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200,
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
print('y_test')
print(y_test)
print('y_train')
print(y_train)

print(plt.plot(y_test[1:200]))
print(plt.plot(y_pred_RF[1:200]))
plt.title("Figure of Random Forest")
print(plt.show())
print(plt.scatter(y_test,y_pred_RF))
plt.title("RF Scatter Plot")
print(plt.show())

##Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF)
MBE_RF=np.mean(y_test-y_pred_DT) #here we calculate MBE
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
NMBE_RF=MBE_RF/np.mean(y_test)
print('Random forest regression results')
print(MAE_RF,MBE_RF,MSE_RF,RMSE_RF,cvRMSE_RF,NMBE_RF)

##Results were best with random forest method so that will be used as final method for the model. Yay.


import pickle
with open('RF_model.pkl','wb') as file:
    pickle.dump(RF_model, file)

