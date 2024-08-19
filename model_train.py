import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

# 读取特征数据和标签
df = pd.read_excel('dataset.xlsx')
data = df.iloc[:, 0:5]  # index location 索引位置
labels = df.iloc[:, 6]
X= data.values
y = labels.values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
data_train = lgb.Dataset(X_train, y_train)

'''def lgb_cv(learning_rate,n_estimators,min_child_weight, colsample_bytree , min_child_samples,
           num_leaves, subsample, max_depth,reg_alpha, reg_lambda):
    model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',
         learning_rate=float(learning_rate),
           n_estimators=int(n_estimators), n_jobs=-1,
           random_state=None,colsample_bytree=float(colsample_bytree),  min_child_weight=float(min_child_weight),
        num_leaves=int(num_leaves),max_depth=int(max_depth),min_child_samples=int(min_child_samples),
           reg_alpha=float(reg_alpha), reg_lambda=float( reg_lambda),
           subsample=float(subsample))
    cv_score = cross_val_score(model, X_train, y_train, scoring="f1", cv=5).mean()
    return cv_score
# 使用贝叶斯优化
lgb_bo = BayesianOptimization(
        lgb_cv,
        {'learning_rate':(0.001,0.3),
         'n_estimators':(10,1000),
         'min_child_weight': (0.0001, 0.5),
         'reg_alpha': (0, 5),
         'reg_lambda': (0, 5),
         'colsample_bytree': (0.7, 1),
         'min_child_samples': (2, 25),
         'num_leaves': (5, 250),
         'subsample': (0.7, 1),
         'max_depth': (2, 10)

         })
lgb_bo.maximize()
lgb_bo.max
'''

# 将优化好的参数带入进行使用
model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',
           colsample_bytree=0.8218, learning_rate=0.2332,max_depth=9,
           min_child_samples=20, min_child_weight=0.4957,
           n_estimators=519, num_leaves=70,reg_alpha=0.6304, reg_lambda =3.487,
           subsample=0.7411,n_jobs=-1, random_state=None
           )
cv_score = cross_val_score(model, X_val, y_val, scoring="f1", cv=5).mean()
cv_score
model.fit(X_train, y_train)
y_pred=model.predict(X_val)
print('accuracy_score:',accuracy_score(y_pred,y_val))
print('precision_score:',precision_score(y_pred,y_val))
print('f1_score:',f1_score(y_pred,y_val))
print('recall_score:',recall_score(y_pred,y_val))
joblib.dump(model, 'LGBClassifier.pkl')

