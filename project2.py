import pandas as pd
daimond = pd.read_csv(r'C:\Users\hp\Desktop\diamonds.csv')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print(daimond.head())
#print(daimond.describe())
#print(daimond.info())
# /// we try to explore the data thruogh relations
#sns.pairplot(daimond,kind='scatter' )
#sns.jointplot(x='carat',y='price',data=daimond,kind='scatter')
#sns.lmplot(x='carat',y='price',data=daimond)
#plt.show()
#/// we try to convert catogorical features
#cut = pd.get_dummies(daimond['cut'],drop_first=True)
#color = pd.get_dummies(daimond['color'],drop_first=True)
#clarity = pd.get_dummies(daimond['clarity'],drop_first=True)
#print(cut )
#print(clarity)
#print(color)
# // now we add our cols to the data and drop the old ones note we didnt drop in place so our data doesnt get defected ///// we did do it inplace sence we dont mind our changes to the data
#daimond.drop(['cut','clarity','color'],axis=1,inplace=True)
#daimond = pd.concat([daimond,cut,clarity,color],axis=1)
#print(daimond.head())
# /// now we should be able to plot our new data in a clear way and if that works we can then start the test and split method
#sns.pairplot(daimond,kind='scatter' )
#plt.show()
#/// unfortunattly our pairplot is messy and hard to read so we try another method to convert the catogorical values into numarical by assiging each catogoty to a number
print(daimond["cut"].unique())
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
cut = daimond["cut"].map(cut_map)
daimond["cut"] = cut
print(daimond["cut"])

print(daimond["color"].unique())
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
color = daimond["color"].map(color_map)
daimond["color"] = color
print(daimond["color"])

print(daimond["clarity"].unique())
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
clarity = daimond["clarity"].map(clarity_map)
daimond["clarity"] = clarity
print(daimond["clarity"])
#sns.pairplot(daimond,kind='scatter' )
#plt.show()
print(daimond.columns)
print(daimond.head())
#//// training the data
x = daimond.drop('price',axis=1)
y = daimond['price']
from sklearn.model_selection import train_test_split
x_train,x_test , y_train , y_test = train_test_split(x,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['coefficients'])
print(coeff_df)
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions), bins=30)
plt.show()
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.square(metrics.mean_squared_error(y_test,predictions)))
#//// risiduals
coeff_dataframe = pd.DataFrame(lm.coef_,x.columns,columns=['coeff'])
print(coeff_dataframe)


