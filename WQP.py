import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')

path = "C:/Users/User/Desktop/wineQuality.csv"
wine = pd.read_csv(path)
correlations = wine.corr()['quality'].drop('quality')
print(correlations)
print(wine.describe())
print(wine.groupby('quality').size())


scatter_matrix(wine)
plt.show()

wine.hist()
plt.show()
def get_features (correlation_threshold):
    abs_corrs=correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

features=get_features(0.05)
print (features)
x=wine[features]
y=wine['quality']

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state =3)

reg=LinearRegression()
reg.fit(x_train, y_train)
print(reg.coef_)
train_pred = reg.predict(x_train)
test_pred=reg.predict(x_test)
print(test_pred)
print("Accuracy due to linear regression test is :(:.2f)",format(reg.score(x_train,y_train)))

Lreg=LogisticRegression()
Lreg.fit(x_train, y_train)
y_pred=Lreg.predict(x_test)
print("Accuracy of logistic regression classifier on test set:(:.2f)",format(Lreg.score(x_train,y_train)))


train_rmse = mean_squared_error(train_pred, y_train)**0.5
print(train_rmse)
test_rmse=mean_squared_error(test_pred, y_test)**0.5
print (test_rmse)
# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
print (predicted_data)
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,test_pred))
print('Mean_Squared_Error:', metrics.mean_squared_error(y_test,test_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,test_pred)))
#displaying coefficients of each feature
coefficients=pd.DataFrame(reg.coef_,features)
coefficients.columns = ['Coefficient']
print(coefficients)
