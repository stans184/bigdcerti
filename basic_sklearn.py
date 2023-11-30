""" 기억이 안 날 경우 """
# dir / __all__ / help 활용해야 한다
# print(dir(sklearn))
# print(sklearn.__all__)
# print(help(train_test_split))


"""
작업형 2 flow
1. shape로 크기 파악
2. target값 분포 파악
3. info()로 데이터 형태 파악
4. isnull().sum() / isna().sum() 결측치 파악
5. describe() 수치형 이상치 파악
6. object 자료형 처리
7. model seperation
8. ML model
    - Classifier : RandomForest, lightgbm
    - Regressor : RandomForest, xgboost, lightgbm
9. Evaluation
"""


""" scaling """
# 민-맥스 스케일링 MinMaxScaler (모든 값이 0과 1사이)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

# 표준화 StandardScaler (Z-score 정규화, 평균이 0 표준편차가 1인 표준 정규분포로 변경)
# 주어진 데이터에 이상치가 없고, 정규분포를 따를 때
from sklearn.preprocessing import StandardScaler

# 로버스트 스케일링 : 중앙값과 사분위 값 활용, 이상치 영향 최소화 장점
# 주어진 데이터에 이상치가 많을 때
# 이상치가 있을 때, MinMaxScaler와 StandardScaler는 변환이 별로 이쁘게 안됨
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale

scaler = RobustScaler()
# 학습시켜야 하는 training data
# 들어가는 데이터는 DataFrame 형태여야 한다
# ex1) df['qsec'].values.reshape(-1,1)
# ex2) df[['qsec']]
# series 형태로 바로 처리가 가능한 minmax_scale 과 같은 함수를 알아두는 것도 좋은 방법
# df['qsec'] = minmax_scale(df['qsec'])
scaler.fit_transform()
# 제출해야 하는 test data
scaler.transform()


""" log 변환 """
# 특정 구간으로 몰려 있는 데이터가 있을 때
# 로그 변환을 통해 정규분포화 시킬 수 있다
import numpy as np
np.log1p()
# np.exp : log 변환한 데이터를 다시 원래 데이터로 돌리기
np.exp(np.log1p(X_train['fnlwgt'])).hist()


""" Encoding """
# 강사의 생각 : 카테고리가 10개 미만 - one-hot encoding / 10개 이상 - LabelEncoding
# 각 범주의 성향을 파악하고 하면 더 좋음
"""
1. train 과 test 모두 category가 같다면 
    > one-hot encoding과 labelencoding 모두 상관없음
2. train의 범주가 test의 범주를 포함한다
    > LabelEncoding 하거나, 두 범주를 합쳐서(pd.concat) one-hot encoding
3. test의 범주가 train을 포함하거나, 카테고리가 좀 다르다면
    > 두 범주를 합쳐서(pd.concat) Encoding을 진행하고 다시 분리
"""
## category로 변경 (lightgbm)
## Series.astype('category').cat.codes
train['Gender'] = train['Gender'].astype('category').cat.codes

# Label Encoding
# 1,2,3... 등의 숫자로 구분시키는 것
# 라벨 인코딩은 각 범주를 정수로 매핑하기 때문에 범주 간의 상대적인 크기를 고려합니다. 
# 이는 순서가 있는 범주형 데이터에 유용합니다.
# 단일 열에 적용 가능: LabelEncoder는 주로 단일 열에 적용되며, 해당 열의 범주를 정수로 매핑합니다.
# 선형 모델에서 유용: 선형 모델과 같이 순서가 있는 범주형 데이터를 고려하는 모델에서 라벨 인코딩이 유용할 수 있습니다.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# 학습시켜야 하는 training data
le.fit_transform()
# 제출해야 하는 test data
le.transform()

# One-Hot Encoding
# 결과값이 0 혹은 1이 나오는 encoding
# 고유값이 너무 많다면, label encoding보다는 one hot encoding이 적합
# 범주 간의 순서를 고려하지 않습니다: 범주 간에 상대적인 크기가 중요하지 않을 때 사용됩니다.
# 다중 열에 적용 가능: pd.get_dummies와 같은 함수를 사용하여 여러 범주형 열에 동시에 적용할 수 있습니다.
# 트리 기반 모델에서 유용: 결정 트리와 같은 트리 기반 모델에서 범주형 데이터를 다루는 데에 원핫 인코딩이 유용할 수 있습니다.
import pandas as pd
# One_Hot Encoding
train = pd.get_dummies(train, columns=cols)
test = pd.get_dummies(test, columns=cols)


""" 결측치를 채우는 class """
# 결측치의 파악에는 domain knowledge 중요
# 결측치 삭제는 왠만하면 하지 말길, 리스크가 있다
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)


""" 검증용 데이터 분리하기 """
# test_size를 통해 train과 test를 어느정도 비율로 나눌 것인지 확인
# random_state를 지정해서 결과 일관성 유지 / 디버깅 용이
from sklearn.model_selection import train_test_split
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y, test_size=0.2, random_state=2023)


""" ML model """
# ML model을 학습시키기 전에, 전처리를 하는 과정이 필요하다
# 모든 feature에 대해서 engineering 후 투입하는 것보다, 기초적인 결측치 처리만 수행한 baseline을 기준으로 option을 넣어가며 결과 비교

# Logistic Regression
# 일종의 Classifier, 범주형 종속변수
from sklearn.linear_model import LogisticRegression

# Linear Regression
# Regressior, 연속형 종속변수
from sklearn.linear_model import LinearRegression

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# RandomForest
# 분류 나무
from sklearn.ensemble import RandomForestClassifier
# 회귀 나무
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=2023, max_depth=3, n_estimators=100)
# random_state : ML model을 정해주는 것, 설정해주지 않으면 매번 다른 모델을 불러오게 된다
# 학습시키기
rf.fit(X_train, y_train)
# 결과값 예측
# predict : 결과 클래스를 보여줌
rf.predict(X_test)
# predict_proba : 결과 클래스가 나올 확률
rf.predict_proba(X_test)
# predict_log_proba : 결과 클래스 확률의 log 값

# XGBoost
from xgboost import XGBClassifier
from xgboost import XGBRegressor
xgb = XGBClassifier(random_state=2023, max_depth=3, n_estimators=100, learning_rate=0.02)
# random_state : ML model을 정해주는 것, 설정해주지 않으면 매번 다른 모델을 불러오게 된다
# max_depth : tree의 깊이를 설정해주는 hyper parameter (3, 5, 7 ... ~ 12)
# n_estimators : tree의 개수를 설정해주는 hyper parameter (100, 200, 400.. ~1000)
# learning_rate : 학습률을 설정해주는 hyper parameter, n_estimators와 같이 조정해주어야 한다
# tree의 개수가 많으면 learning_rate를 낮추는 방식으로 (0.1 ~ 0.01)
# Regression 모델은 최근 대부분 xgboost 혹은 lightgbm 으로 사용
# Regression 문제에서는 outlier 처리를 잘 해줘야 한다
# Regression 문제 상황에서, 다중 공산성으로 인한 해석력 감소가 있을 수 있다

# lightGBM
# 범주형 데이터를 onehot / label encoding이 필요없음?
# object type을 category형 type으로 바꿔주면, lightgbm이 알아서 변경해줌
train['주구매상품'] = train['주구매상품'].astype('category')
# 결측치를 처리 안해도 되는데.. 이건 상황에 따라서
# 10000 rows 보다 적은 데이터들에 사용하면 overfitting 우려가 있음 (경험적으로)
# hyper parameter에 민감하다, 특히 max_depth에
import lightgbm as lgb
model = lgb.LGBMClassifier(random_state=2023, max_depth=5, n_estimators=200, learning_rate=0.1)
model = lgb.LGBMRegressor(random_state=2023, max_depth=3, n_estimators=400, learning_rate=0.05)
model.fit(X_tr, y_tr)
model.predict(X_val)
model.predict_proba(X_val)


""" hyper-parameter 튜닝 시 overfitting 주의!! """
# 너무 깊은 tree로 갈수록, tree의 개수가 너무 커질수록 overfitting 위험
# max_depth : tree의 깊이를 설정해주는 hyper parameter (3, 5, 7 ... ~ 12)
# n_estimators : tree의 개수를 설정해주는 hyper parameter (100, 200, 400.. ~1000)


""" Evaluation """
""" 회귀 모델 평가 지표 """
# MSE(Mean Squared Error)
# error : 낮을수록 좋음
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)

# MAE(Mean Absolute Error)
# error : 낮을수록 좋음
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)

# RMSLE
from sklearn.metrics import mean_squared_log_error
print(mean_squared_log_error() ** 0.5)

# MAPE
from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error())

# 결정 계수(R-squared) ***
# 높을수록 좋음
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

# RMSE(Root Mean Squared Error) ***
# error : 낮을수록 좋음
rmse = mse ** 0.5

""" 분류 모델 평가 지표 """
# 정확도 : 높으면 좋음
from sklearn.metrics import accuracy_score
# 2진 분류 모델에서 사용된다
# 양성 클래스와 음성 클래스의 확률이 얼마나 정확한지 비교
accuracy_score(y_true=, y_pred=)

# 재현율 (민감도)
from sklearn.metrics import recall_score
print(recall_score())

# roc auc score : 확률을 비교, 높으면 좋음
# predict_proba로 나온 결과 중, 1이 나올 확률로 비교해야 함
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true=, model.predict_proba[:, 1])

# F1 스코어(F1 Score) *** 필수암기, 높으면 좋음
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
# 다중 클래스 분류에서 사용할 땐, pos_label 을 하나의 변수로 특정해줌
# 양성 클래스를 'A'로 설정해주는 것
f1 = f1_score(y_true_str, y_pred_str, pos_label='A')

# 다중 분류 모델은, 실제 데이터와 예측 데이터의 column이 동일한 형태를 가지고 있는지 확인해야 한다
# F1 스코어(F1 Score) ***
f1 = f1_score(y_true, y_pred, average='macro')  # average= micro, macro, weighted
f1 = f1_score(y_true_str, y_pred_str, average='macro')

# 수치형일 때는 자동으로 원핫인코딩에서 제외함. 컬럼 지정 필요
y_true_onehot = pd.get_dummies(y_true[0])
# 인코딩된 순서와 확률 컬럼 순서가 같인지 확인
print("y_true의 원-핫 인코딩된 컬럼 순서:", y_true_onehot.columns)
print("y_pred_proba의 컬럼 순서:", y_pred_proba.columns)
# 'ovo'는 클래스의 쌍이 많을 경우에는 계산이 많이 필요하지만, 
# 'ovr'은 클래스가 많더라도 각 이진 분류 문제가 독립적으로 해결되기 때문에 더 빠름
# 'ovr'은 클래스 간 불균형이 심한 경우에 사용될 수 있습니다.
# 'ovo' 방식
roc_score_ovo = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovo')
print("ROC AUC (OvO):", roc_score_ovo)
# 'ovr' 방식
roc_score_ovr = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr')
print("ROC AUC (OvR):", roc_score_ovr)


""" Hyper parameter tuning """
"""
[max_depth] : default -1
tree의 깊이를 설정해주는 hyper parameter
너무 깊어지면 overfitting 우려가 있음
3, 5, 7.. ~ 12 로 증가됨

[n_estimators] : default 100
tree의 개수를 설정해주는 hyper parameter
너무 많아지면 계산 속도가 느려짐
100, 200, 400.. ~1000 으로 증가됨

[learning_rate] : xgboost, lightgbm, default 0.1
학습률을 설정해주는 hyper parameter
n_estimatorsd와 같이 조정해주어야 한다
tree의 개수가 많으면 learning_rate를 낮추는 방식으로
0.1 ~ 0.01

1. human search
2. Grid search (GridSearchCV) : 주어진 HP의 조합을 모두 돌려보는 방식.
3. Bayesian Optimization(hyperopt, optuna ...) : hyper-parameter를 최적화하는 베이지안 방식 사용
"""
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth' : [-1, 3, 5],
    'n_estimators' : [50, 100, 200],
    'learning_rate' : [0.1, 0.01]
}

gvc = GridSearchCV(param_grid=param_grid, model, scoring='neg_mean_squared_error')
gvc.fit()
gvc.cv_results_
gvc.best_estimator_
gvc.best_params_