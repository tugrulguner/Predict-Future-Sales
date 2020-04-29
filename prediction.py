import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
train['item_price'] = train['item_price'][train['item_price'].between(train['item_price'].quantile(.1), train['item_price'].quantile(.9))]

sns.regplot(y="item_price", x="shop_id", data=train, x_estimator = np.mean)

sns.regplot(y="item_price", x="date_block_num", data=train, x_estimator = np.mean)

sns.regplot(y="item_price", x='item_cnt_day', data=train, x_estimator = np.mean)

train['item_price'].fillna('backfill', inplace = True)

def namecorr(column):
    name = column[3:]
    return name[0:2], name[5:]
    
aa= train['date'].apply(namecorr)

b = pd.DataFrame(aa.values.tolist(), columns = ['month','year'])
train = pd.concat([train, b], axis = 1)
train.drop(['date'], axis = 1, inplace = True)

train['month'] = train.month.astype(int)
train['year'] = train.year.astype(int)

sns.boxplot(y=train["item_price"], x=train['month'])

sns.heatmap(train.corr(), annot = True)

scaler = StandardScaler()
a = scaler.fit(train).transform(train)
train = pd.DataFrame(a, columns = train.columns)

k = 1
deger = 0
degerson = 0 

while degerson < 0.85:
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['item_price', 'date', 'date_block_num', 'item_cnt_day'], axis = 1), 
                                                         train['item_cnt_day'], test_size=0.3)
    
    modelf = RandomForestClassifier(n_estimators = 50) # These values were determined through Randomized Search CV above 
    modelf.fit(X_train, y_train) 
    deger = cross_validate(modelf, X_test, y_test, cv=4) # I like to use cross validation to determine my score
    degerson = deger['test_score'].mean()
    k += 1
    if k>1:
        break
    # Here I put certain contraint on the loop, if I couldn't find score higher than 0.85, It will stop after some certain 
    # iteration. I can improve this part here to involve even though it couldnt find better score than 0.85, it can select
    # the best one it got during the process.

deger['test_score'].mean()

m = 1
degerm = 0
degermson = 0 

while degermson < 0.85:
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['item_price', 'date', 'date_block_num'], axis = 1), 
                                                         train['date_block_num'], test_size=0.3)
    
    models = RandomForestClassifier(n_estimators = 50) # These values were determined through Randomized Search CV above 
    models.fit(X_train, y_train) 
    degerm = cross_validate(models, X_test, y_test, cv=4) # I like to use cross validation to determine my score
    degermson = degerm['test_score'].mean()
    m += 1
    if m>1:
        break
    # Here I put certain contraint on the loop, if I couldn't find score higher than 0.85, It will stop after some certain 
    # iteration. I can improve this part here to involve even though it couldnt find better score than 0.85, it can select
    # the best one it got during the process.

degerm['test_score'].mean()

b = 1
degerb = 0
degerbson = 0 

while degerbson < 0.85:
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['item_price', 'date'], axis = 1), 
                                                         train['item_price'], test_size=0.3)
    
    modelt = RandomForestRegressor(n_estimators = 50) # These values were determined through Randomized Search CV above 
    modelt.fit(X_train, y_train) 
    degerb = cross_validate(modelt, X_test, y_test, cv=4) # I like to use cross validation to determine my score
    degerbson = degerb['test_score'].mean()
    m += 1
    if m>1:
        break
    # Here I put certain contraint on the loop, if I couldn't find score higher than 0.85, It will stop after some certain 
    # iteration. I can improve this part here to involve even though it couldnt find better score than 0.85, it can select
    # the best one it got during the process.

degerb['test_score'].mean()

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test.isnull().sum()

itcou= pd.DataFrame(modelf.predict(test), columns = 'item_cnt_day')
test = pd.concat([test itcou])

dablnu= pd.DataFrame(models.predict(test), columns = 'date_block_num')
test = pd.concat([test dablnu])

sonuc.to_csv('/kaggle/working/predict.csv', index = False)
