""" 기억이 안 날 경우 """
# dir / __all__ / help 활용해야 한다
# print(dir(sklearn))
# print(sklearn.__all__)
# print(help(train_test_split))


""" scaling """
# 민-맥스 스케일링 MinMaxScaler (모든 값이 0과 1사이)
from sklearn.preprocessing import MinMaxScaler

# 표준화 StandardScaler (Z-score 정규화, 평균이 0 표준편차가 1인 표준 정규분포로 변경)
from sklearn.preprocessing import StandardScaler

# 로버스트 스케일링 : 중앙값과 사분위 값 활용, 이상치 영향 최소화 장점
# 이상치가 있을 때, MinMaxScaler와 StandardScaler는 변환이 별로 이쁘게 안됨
from sklearn.preprocessing import RobustScaler

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
# Label Encoding
# 1,2,3... 등의 숫자로 구분시키는 것
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# 학습시켜야 하는 training data
le.fit_transform()
# 제출해야 하는 test data
le.transform()

# One-Hot Encoding
# 결과값이 0 혹은 1이 나오는 encoding
import pandas as pd
# One_Hot Encoding
train = pd.get_dummies(train, columns=cols)
test = pd.get_dummies(test, columns=cols)

""" ML model """
# ML model을 학습시키기 전에, 전처리를 하는 과정이 필요하다
# 모든 feature에 대해서 engineering 후 투입하는 것보다, 기초적인 결측치 처리만 수행한 baseline을 기준으로 option을 넣어가며 결과 비교

""" 검증용 데이터 분리하기 """
# test_size를 통해 train과 test를 어느정도 비율로 나눌 것인지 확인
# random_state를 지정해서 데이터들을 매번 다르게 변환해주는 것도 좋은 작업임
from sklearn.model_selection import train_test_split
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y, test_size=0.2, random_state=2023)


""" LinearRegression """
from sklearn.linear_model import LinearRegression

""" RandomForest """
# 분류 나무
from sklearn.ensemble import RandomForestClassifier
# 회귀 나무
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=2023, max_depth=3, n_estimators=100)
# random_state : ML model을 정해주는 것, 설정해주지 않으면 매번 다른 모델을 불러오게 된다
# max_depth : tree의 깊이를 설정해주는 hyper parameter (3, 5, 7 ... ~ 12)
# n_estimators : tree의 개수를 설정해주는 hyper parameter (100, 200, 400.. ~1000)

# 학습시키기
rf.fit(X_train, y_train)
# 결과값 예측
# predict : 결과 클래스를 보여줌
rf.predict(X_test)
# predict_proba : 결과 클래스가 나올 확률
rf.predict_proba(X_test)
# predict_log_proba : 결과 클래스 확률의 log 값


""" XGBoost """
from xgboost import XGBClassifier
from xgboost import XGBRegressor
xgb = XGBClassifier(random_state=2023, max_depth=3, n_estimators=100, learning_rate=0.02)
# random_state : ML model을 정해주는 것, 설정해주지 않으면 매번 다른 모델을 불러오게 된다
# max_depth : tree의 깊이를 설정해주는 hyper parameter (3, 5, 7 ... ~ 12)
# n_estimators : tree의 개수를 설정해주는 hyper parameter (100, 200, 400.. ~1000)
# learning_rate : 학습률을 설정해주는 hyper parameter, n_estimators와 같이 조정해주어야 한다
# tree의 개수가 많으면 learning_rate를 낮추는 방식으로 (0.1 ~ 0.01)

""" 회귀 모델 평가 지표 """
# MSE(Mean Squared Error)
# error : 낮을수록 좋음
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)

# MAE(Mean Absolute Error)
# error : 낮을수록 좋음
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)

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

# roc auc score : 확률을 비교, 높으면 좋음
# predict_proba로 나온 결과 중, 1이 나올 확률로 비교해야 함
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true=, y_score=)

# F1 스코어(F1 Score) *** 필수암기, 높으면 좋음
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
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
# 'ovo' 방식
roc_score_ovo = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovo')
print("ROC AUC (OvO):", roc_score_ovo)
# 'ovr' 방식
roc_score_ovr = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr')
print("ROC AUC (OvR):", roc_score_ovr)