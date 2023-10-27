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
# random_state 설정을 해주어야 특정한 모델을 가지고 지속적인 사용한다는 뜻이다
rf = RandomForestRegressor(random_state=2023)
# 학습시키기
rf.fit(X_train, y_train)
# 결과값 예측
# predict
# 결과 클래스를 보여줌
rf.predict(X_test)
# predict_proba
# 결과 클래스가 나올 확률
# predict_log_proba
# 결과 클래스 확률의 log 값


""" XGBoost """
from xgboost import XGBClassifier
from xgboost import XGBRegressor


""" MSE, RMSE """
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true="", y_pred="")

""" 평가 모델 """
# 정확도
from sklearn.metrics import accuracy_score
# 2진 분류 모델에서 사용된다
# 양성 클래스와 음성 클래스의 확률이 얼마나 정확한지 비교
from sklearn.metrics import roc_auc_score