#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from scipy.stats import norm, skew
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.shape


# In[3]:


print(train.describe())


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[8]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[9]:


sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()
res = stats.probplot(train['SalePrice'], plot=plt)


# In[11]:


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(), cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()


# In[17]:


important_num_cols = list(train.corr()["SalePrice"][(train.corr()["SalePrice"]>0.50) | (train.corr()["SalePrice"]<-0.50)].index)
cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]
important_cols = important_num_cols + cat_cols
train = train[important_cols]
train.drop('SalePrice',axis = 1,inplace = True)
train.info()


# In[45]:


print("Missing Values by Column")
print("-"*30)
print(train.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",train.isna().sum().sum())


# In[46]:


test = test[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea','MSZoning',
             'Utilities','BldgType','Heating','KitchenQual','SaleCondition','LandSlope']]


# In[47]:


test


# In[48]:


train


# In[49]:


x = train.drop(['SalePrice'],axis=1)
y = train.loc[:,'SalePrice']
x = pd.get_dummies(x, columns=cat_cols)


# In[50]:


important_num_cols.remove('SalePrice')
scaler = StandardScaler()
x[important_num_cols] = scaler.fit_transform(x[important_num_cols])


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
# models = pd.DataFrame(columns=["Model","MAE","MSE","RMSE","R2 Score","RMSE(Cross-Validation)"])


# In[60]:


#Cross validation
kf = KFold(n_splits=12, random_state=42, shuffle=True)
scores = {}


# In[72]:


#MSE
def rmsle(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def cv_rmse(model, X=x_train , Y=y_train):
    rmse = np.sqrt(-cross_val_score(model,X, Y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[52]:


#Linear Regression
LR_reg = LinearRegression()
LR_reg.fit(x_train,y_train)
Price_predictions = LR_reg.predict(x_test)

R2_score = r2_score(y_test,Price_predictions)
print("Linear Regression's R2 Score: ",R2_score)


# In[89]:


#Polynomial Regression
PR_reg = PolynomialFeatures(degree=2)
# x_train_poly = scaler.fit_transform(x_train)
# x_test_poly = scaler.fit_transform(x_test)
PLR_reg = LinearRegression()
PLR_reg.fit(x_train,y_train)
Price_predictions = PLR_reg.predict(x_test)
R2_score = r2_score(y_test,Price_predictions)
print("Poly Polynomial Regression's R2 Score: ",R2_score)


# In[26]:


#RandomForest
RD = RandomForestRegressor()
RD.fit(x_train,y_train)
Price_predictions = RD.predict(x_test)
R2_score = r2_score(y_test,Price_predictions)
print("Random Forest's R2 Score: ",R2_score)


# In[80]:


#DecisionTree
DT = DecisionTreeRegressor()
DT.fit(x_train,y_train)
Price_predictions = DT.predict(x_test)
R2_score = r2_score(y_test,Price_predictions)
print("Decision Tree's R2 Score: ",R2_score)


# In[91]:


# XGBoost Regressor
xgboost = XGBRegressor()
xgboost.fit(x_train,y_train)
Price_predictions = xgboost.predict(x_test)
R2_score = r2_score(y_test,Price_predictions)
print("XGboost's R2 Score: ",R2_score)


# In[106]:


# Support Vector Regressor
SVM=SVC(kernel="linear", decision_function_shape="ovo")
SVM.fit(x_train,y_train)
Price_predictions = SVM.predict(x_test)
R2_score = r2_score(y_test,Price_predictions)
print("SVM's R2 Score: ",R2_score)


# In[61]:


decision_tree_model = DecisionTreeRegressor()
score = cv_rmse(decision_tree_model)
print("Decision Tree Model: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[62]:


# Random Forest Regressor
random_forest_model = RandomForestRegressor(random_state=42)
score = cv_rmse(random_forest_model)
print("Random Forest Model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['random_forest'] = (score.mean(), score.std())


# In[63]:


# XGBoost Regressor
xgboost = XGBRegressor(objective='reg:squarederror',random_state=42)
score = cv_rmse(xgboost)
print("xgboost_model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Xgboost'] = (score.mean(), score.std())


# In[70]:


# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
score = rmsle(svr)
print("Support Vector Machine: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Support Vector Machine'] = (score.mean(), score.std())


# In[79]:


decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(x_train,y_train)
decision_tree_model = decision_tree_model.predict(x_test)
score = rmsle(y_test,decision_tree_model)
print(score)


# In[ ]:




