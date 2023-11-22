""" 작업형 3 """
"""
basic theory

귀무가설 : 기존과 차이가 없다
대립가설 : 연구자가 입증하고 싶은 새로운 사실

결과
- 검정통계량 : 주어진 데이터와 귀무가설 간의 차이를 통계적으로 나타낸다
- p-value (유의수준 0.05)
    - 0.05보다 작으면, 대립가설 채택, 귀무가설 기각
    - 0.05보다 크면, 귀무가설 채택, 대립가설 기각

검정 flow

[단일 검정 (모집단 1개, 한가지 조건을 걸고 검정)]
1. 샤피로-월크 검정 (정규성 확인) : shapiro(data)
    1) 정규성 만족  : 단일 표본 검정 : ttest_1samp(data)
    2) 정규성 불만족 : 윌콕슨 검정 : wilcoxon(data)
    
[대응 검정 (모집단 1개, event 전후 검정)]
1. 샤피로-윌크 검정 (정규성 확인) : shapiro(diff) diff = A - B
    1) 정규성 만족  : 대응 표본 검정 : ttest_rel(A, B)
    2) 정규성 불만족 : 윌콕슨 검정 : wilcoxon(A, B)

[독립 표본 검정 (모집단 2개)]
1. 샤피로 윌크 검정 (각 집단의 정규성 확인) : shapiro(A), shapiro(B)
    1) 두 집단 모두 정규성 만족 -> 등분산 검정 : levene(A, B)
        (1) 두 집단이 등분산   : 독립 표본 검정 ttest_ind(A, B)
        (2) 두 집단이 다른 분산 : 독립 표본 검정 ttest_ind(A, B, equal_var=False)
    2) 정규성 불만족 : Mann-whitney U 검정 : mannwhitneyu(A, B)
"""

# scipy version 확인
# 시험 환경의 version : 1.7
# 시험 환경에서는 자유도가 나오지 않는다
import scipy
print(scipy.__version__)

""" 단일 표본 검정 : 정규분포일 때 """
# p-value < 0.05 : 귀무가설 기각, 대립가설 채택
# p-value > 0.05 : 귀무가설 채택, 대립가설 기각
# df : 관측치 - 1
# alternative : 양측검정, 단측검정을 결정하는 parameter
# 기본적으로 alternative의 설정값은 two-sided (양측 검정, 크거나, 작거나)
# 'greater' or 'less' : 큰 값, 작은 값 단측 검정
from scipy import stats
print(stats.ttest_1samp(df['무게'], 120, alternative='two-sided'))
# TtestResult(statistic=2.153709967150663, pvalue=0.03970987897788578, df=29)

""" 샤피로 윌크 검정 : 정규성 만족 여부 확인 """
# 귀무가설 (H0): 주어진 데이터 샘플은 정규 분포를 따른다.
# 대립가설 (H1): 주어진 데이터 샘플은 정규 분포를 따르지 않는다.
# Shapiro-Wilk(샤피로-윌크) 정규성 검정
# p-value < 0.05 : 대립가설 채택, 정규분포를 따르지 않는다
# p-value > 0.05 : 귀무가설 채택, 정규분포를 따른다
from scipy import stats
stats.shapiro(df['무게'])

""" 윌콕슨 검정 (비모수 검정)) : 정규성을 따르지 않는 데이터에 대해 """
# 귀무가설 (H0): μ = μ0, "합격 원두(dark)" 상품의 평균 무게는 120g이다.
# 대립가설 (H1): μ < μ0, "합격 원두(dark)" 상품의 평균 무게는 120g 보다 작다
# Wilcoxon(윌콕슨)의 부호 순위 검정 수행
# p-value < 0.05 : 대립가설 채택
# p-value > 0.05 : 귀무가설 채택
from scipy import stats
stats.wilcoxon(df['무게'] - 120, alternative='less')


""" 대응 표본 검정 : 정규분포를 따를 때 """
# μd = (before – after)의 평균
# 귀무가설: μd ≥ 0, 새로운 교육 프로그램은 효과가 없다.
# 대립가설: μd < 0, 새로운 교육 프로그램은 효과가 있다.

# 대응표본검정
# alternative : 앞의 데이터가, 뒤의 데이터에 비해서 어떠한가
# 전, 후 데이터의 위치를 잘 넣어야 함
from scipy import stats
print(stats.ttest_rel(df['before'], df['after'], alternative='less'))
print(stats.ttest_rel(df['after'], df['before'], alternative='greater'))

""" 샤피로 윌크 검정 : 정규성을 만족하는지 확인 """
# 귀무가설 (H0): 주어진 데이터 샘플은 정규 분포를 따른다.
# 대립가설(H1): 주어진 데이터 샘플은 정규 분포를 따르지 않는다.
# Shapiro-Wilk(샤피로-윌크) 정규성 검정
# p-value < 0.05 : 대립가설 채택, 정규분포를 따르지 않는다
# p-value > 0.05 : 귀무가설 채택, 정규분포를 따른다
from scipy import stats
# 더하고 빼는 순서는 상관이 없다
df['diff'] = df['after'] - df['before']
stats.shapiro(df['diff'])

""" 윌콕슨 비모수 검정 : 정규성을 만족하지 않는 데이터에 대해 """
# 귀무가설 (H0): μ = μ0, "합격 원두(dark)" 상품의 평균 무게는 120g이다.
# 대립가설(H1): μ < μ0, "합격 원두(dark)" 상품의 평균 무게는 120g 보다 작다
# Wilcoxon(윌콕슨)의 부호 순위 검정 수행
# p-value < 0.05 : 대립가설 채택
# p-value > 0.05 : 귀무가설 채택
from scipy import stats
stats.wilcoxon(df['after'], df['before'], alternative='greater')
# 혹은 diff 로도 진행 가능
df['diff'] = df['after'] - df['before']
stats.wilcoxon(df['diff'], alternative='greater')


""" 독립 표본 검정 : 서로 다른 두 집단에 대해서 """
A = [85, 90, 92, 88, 86, 89, 83, 87,
     84, 50, 60, 39, 28, 48, 38, 28]
B = [82, 82, 88, 85, 84, 74, 79, 69,
     78, 76, 85, 84, 79, 89]

""" 샤피로-윌크 검정 : 정규성 검정 """
# 서로 다른 집단이 정규성읆 만족하는지 각각 검정
# p-value < 0.05 : 대립가설 채택, 정규분포가 아니다
# p-value > 0.05 : 귀무가설 채택, 정규분포이다
from scipy import stats
stats.shapiro(A)
stats.shapiro(B)

""" Levene 검정 : 등분산 검정 """
# 두 집단이 모두 정규성을 만족할 때
# 동일한 분산을 가지고 있는지 검정
# p-value < 0.05 : 대립가설 채택, 두 집단은 분산이 다르다.
# p-value > 0.05 : 귀무가설 채택, 두 집단은 등분산이다.
from scipy import stats
stats.levene(A, B)

""" 독립 표본 검정 진행 """
# 최소한 두 집단 모두 정규성을 만족할 때 진행되어야 함
# 등분산 여부도 확인해서, 그에 따른 옵션 설정 필요
# 분산이 같으면 equal_var True, 다르면 False (Levene 검정 결과에 따라 다름)
# p-value < 0.05 : 대립가설 채택, 두 집단은 평균이 다르다
# p-value > 0.05 : 귀무가설 채택, 두 집단은 평균이 같다
from scipy import stats
stats.ttest_ind(A, B, equal_var=True)

""" Mann-whitneyu 검정 : 정규성을 만족하지 않을 때 """
# 두 집단 중 하나라도 정규성을 만족하지 않을 때
# alternative : 대립가설의 방향을 지정하는 옵션
# alternative='two-sided' : 두 집단의 중앙값은 다르다
# alternative='greater'   : A 중앙값 > B 중앙값
# alternative='less'      : A 중앙값 < B 중앙값
# p-value < 0.05 : 대립가설 채택, 두 집단의 중앙값은 다르다
# p-value > 0.05 : 귀무가설 채택, 두 집단의 중앙값은 같다
from scipy import stats
stats.mannwhitneyu(A, B, alternative="two-sided")



""" 범주형 데이터 분석 """
"""

[적합도 검정]
관찰도수와 기대도수의 차이
** 확률로 대입하면 안되고, 빈도로 대입해야 한다 **

[독립성 검정]
두 변수가 서로 독립적인지 확인 (연관성이 있는지)
교차표 테이블로 만들기 (지문을 해석하거나, raw_data를 변환해서 / pd.crosstab() 활용)

"""

""" 적합도 검정 """
# 지난 3년간 빅데이터 분석기사 점수 분포가 60점 미만: 50%, 60-70점 35%, 80점이상 15% 였다.
# 300명을 대상으로 적용한 결과 60점 미만: 150명, 60-70점: 120명, 80점이상: 30명이었다.
# 유의수준 0.05
# 새로운 시험문제 유형과 기존 시험문제 유형은 점수에 차이가 없는지 검정하시오.
# 귀무가설(H0): 새로운 시험문제는 기존 시험문제 점수와 동일하다.
# 대립가설(H1): 새로운 시험문제는 기존 시험문제 점수와 다르다.
# 관찰값
ob = [150, 120, 30]
# 기대값
ex = [0.5*300, 0.35*300, 0.15*300]

from scipy import stats
stats.chisquare(ob, ex)
# Power_divergenceResult(statistic=7.142857142857142, pvalue=0.028115659748972056)
# p-value < 0.05 : 대립가설 채택, 시험문제는 다르다

""" 독립성 검정 """
# 귀무가설(H0): 언어와 합격 여부는 독립이다.
# 대립가설(H1): 언어과 합격 여부는 독립이지 않다.
# 교차표 데이터
# 주어진 데이터로 직접 table을 만드는 경우 -> 문제를 침착하게 파악하고, 행과 열을 주의해서 맞춰줘야 한다
# R: 합격 80명, 불합격 20명,
# Python: 합격 90명, 불합격 10명
# 1. DataFrame을 만든다면, 행과 열의 방향은 상관 없다
import pandas as pd
df_a = pd.DataFrame(
    {
        "python" : [90, 10],
        "R" : [80, 20]
    },index=['합격', '불합격']
)
df_b = pd.DataFrame(
    {
        "합격" : [90, 80],
        "불합격" : [10, 20]
    }, index=['python', 'R']
)
from scipy import stats
print(stats.chi2_contingency(df_a))
print(stats.chi2_contingency(df_b))

# 2. DataFrame을 만들지 않아도 진행 가능
df = [[80, 20], [90, 10]]
stats.chi2_contingency(df)

# raw_data : 테이블로 만들 수 없다면, pd.crosstab()을 활용
df_c = pd.crosstab(df['언어'], df['합격여부'])
df_d = pd.crosstab(df['합격여부'], df['언어'])



""" 2. 회귀 분석 """
"""
[상관관계 분석]
DataFrame.corr() 읉 통해 상관관계를 알 수 있다
method : pearson (default)
method : spearman
method : kendall

- 특정 변수 간 상관관계 분석도 가능
    df['A'].corr(df['B'])

- 상관관계 t-검정
귀무가설 : 상관 없다
대립가설 : 상관 있다


[단순 선형 회귀 분석]
-> 주어진 데이터로 종속변수, 독립변수를 설정해서 model을 만들어 주어야 한다
-> 종속변수 ~ 독립변수

[다중 선형 회귀 분석]
-> 종속변수 ~ 독립변수1 + 독립변수2 + 독립변수3 ...

[범주형 변수]
-> ols 함수에서 자동으로 one-hot encoding을 진행
-> 다중 공산성이 발생하는 문제가 있고, ols 자체적으로 해결해주기도 함
-> 내가 직접 해결하고 싶다면, pd.get_dummies(df, drop_first=True)

[로지스틱 회귀 분석]
from statsmodels.formula.api import logit
"""

""" 상관관계 분석 """
# method (pearson, spearman, kendall)
df.corr(method='pearson')
df.corr(method='spearman')
df.corr(method='kendall')

# 특정 변수 간 상관관계
df['A'].corr(df['B'])

# 상관관계 t-검정
# 귀무가설 : 상관 없다
# 대립가설 : 상관 있다
from scipy import stats
# 피어슨 검정
stats.pearsonr(df['A'], df['B'])
# 스피어만 검정
stats.spearmanr(df['A'], df['B'])
# 켄달타우 검정
stats.kendalltau(df['A'], df['B'])


""" 단순 선형 회귀 분석 """
# 주어진 데이터로 독립변수, 종속변수 설정을 해주어야 한다
# 종속변수 ~ 독립변수
df = pd.DataFrame({
    '키': [150, 160, 170, 175, 165, 155, 172, 168, 174, 158,
          162, 173, 156, 159, 167, 163, 171, 169, 176, 161],
    '몸무게': [74, 50, 70, 64, 56, 48, 68, 60, 65, 52,
            54, 67, 49, 51, 58, 55, 69, 61, 66, 53]
})

# 선형 회귀 모델 학습
from statsmodels.formula.api import ols
model = ols('키 ~ 몸무게', data=df).fit()

# 학습 결과 출력
model.summary()
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      키   R-squared:                       0.280
Model:                            OLS   Adj. R-squared:                  0.240
Method:                 Least Squares   F-statistic:                     6.984
Date:                Sun, 05 Nov 2023   Prob (F-statistic):             0.0165
Time:                        03:24:15   Log-Likelihood:                -64.701
No. Observations:                  20   AIC:                             133.4
Df Residuals:                      18   BIC:                             135.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    135.8209     11.211     12.115      0.000     112.268     159.374
몸무게            0.4938      0.187      2.643      0.017       0.101       0.886
==============================================================================
Omnibus:                       26.498   Durbin-Watson:                   1.317
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               46.624
Skew:                          -2.181   Prob(JB):                     7.51e-11
Kurtosis:                       9.076   Cond. No.                         464.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# 학습 결과에서 특정 데이터 출력
model.params['몸무게']
model.params['Intercept']
# 결정계수
model.rsquared
# p-value
model.pvalues['몸무게']
model.pvalues['Intercept']

# 학습된 선형 회귀모델로 예측하기
# 몸무게가 50 일때, 예측되는 키
new_data = pd.DataFrame({'몸무게' : [50]})
print(model.predict(new_data))

# 잔차 제곱합
df['residual'] = df['키'] - model.predict(df['몸무게'])
print(sum(df['residual'] ** 2))

# MSE
print((df['residual'] ** 2).mean())
# 혹은
from sklearn.metrics import mean_squared_error
print(mean_squared_error(df['키'], model.predict(df['몸무게'])))

# 예측값에 대한 신뢰구간
prd = model.get_prediction(new_data)
# alpha 설정을 통한 신뢰구간 설정
prd.summary_frame(alpha=0.05)
"""
mean: 예측값
mean_ci_lower ~ mean_ci_upper 신뢰구간

	mean	mean_se	mean_ci_lower	mean_ci_upper	obs_ci_lower	obs_ci_upper
0	160.509227	2.291332	155.695318	165.323136	146.068566	174.949888
"""

""" 다중 선형 회귀 분석 """
df = pd.DataFrame({
    '매출액': [300, 320, 250, 360, 315, 328, 310, 335, 326, 280,
            290, 300, 315, 328, 310, 335, 300, 400, 500, 600],
    '광고비': [70, 75, 30, 80, 72, 77, 70, 82, 70, 80,
            68, 90, 72, 77, 70, 82, 40, 20, 75, 80],
    '플랫폼': [15, 16, 14, 20, 19, 17, 16, 19, 15, 20,
            14, 5, 16, 17, 16, 14, 30, 40, 10, 50],
    '투자':[100, 0, 200, 0, 10, 0, 5, 0, 20, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 플랫폼', data=df).fit()
print(model.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    매출액   R-squared:                       0.512
Model:                            OLS   Adj. R-squared:                  0.454
Method:                 Least Squares   F-statistic:                     8.907
Date:                Sun, 05 Nov 2023   Prob (F-statistic):            0.00226
Time:                        04:14:47   Log-Likelihood:                -108.22
No. Observations:                  20   AIC:                             222.4
Df Residuals:                      17   BIC:                             225.4
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    101.0239     71.716      1.409      0.177     -50.284     252.331
광고비            1.8194      0.807      2.255      0.038       0.117       3.522
플랫폼            5.9288      1.430      4.147      0.001       2.912       8.945
==============================================================================
Omnibus:                       30.534   Durbin-Watson:                   1.354
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               64.655
Skew:                           2.444   Prob(JB):                     9.13e-15
Kurtosis:                      10.327   Cond. No.                         401.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# R 결정계수
model.rsquared
# 회귀계수 (기울기)
# param 과 params의 차이
model.param # 전체
model.param['광고비']
model.param['플랫폼']
model.param['Intercept']
# p-value
model.pvalues['광고비']
model.pvalues['플랫폼']
model.pvalues['Intercept']

# 95% 유의수준
model.conf_int(alpha=0.05)

# 예측값
check = pd.DataFrame({'광고비':[50], '플랫폼':[20]})
model.predict(check)

# 광고비 50, 플랫폼 20일 때, 95% 유의수준
pred = model.get_prediction(check)
pred.summary_frame()
"""
        mean    mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0  310.57033  19.887098     268.612221      352.52844    179.700104   441.440556
"""

""" 범주형 변수 """
df = pd.DataFrame({
    '매출액': [300, 320, 250, 360, 315, 328, 310, 335, 326, 280,
            290, 300, 315, 328, 310, 335, 300, 400, 500, 600],
    '광고비': [70, 75, 30, 80, 72, 77, 70, 82, 70, 80,
            68, 90, 72, 77, 70, 82, 40, 20, 75, 80],
    '플랫폼': [15, 16, 14, 20, 19, 17, 16, 19, 15, 20,
            14, 5, 16, 17, 16, 14, 30, 40, 10, 50],
    '투자':[100, 0, 200, 0, 10, 0, 5, 0, 20, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '유형':['B','B','C','A','B','B','B','B','B','B'
        ,'C','B','B','B','B','B','B','A','A','A']
    })

# one hot encoding, 직접 다중 공산성을 해결하는 parameter
# B와 C가 모두 0인 경우, A를 나타냄
df2 = pd.get_dummies(df, drop_first=True)
print(df2)
"""
	매출액	광고비	플랫폼	투자	유형_B	유형_C
0	300	70	15	100	1	0
1	320	75	16	0	1	0
2	250	30	14	200	0	1
3	360	80	20	0	0	0
4	315	72	19	10	1	0
"""

# ols 자체적으로 해결해주기도 함
from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 유형', data=df)
print(model.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    매출액   R-squared:                       0.720
Model:                            OLS   Adj. R-squared:                  0.667
Method:                 Least Squares   F-statistic:                     13.70
Date:                Sun, 05 Nov 2023   Prob (F-statistic):           0.000110
Time:                        04:23:51   Log-Likelihood:                -102.67
No. Observations:                  20   AIC:                             213.3
Df Residuals:                      16   BIC:                             217.3
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    400.6463     47.477      8.439      0.000     300.000     501.292
유형[T.B]     -160.2695     26.756     -5.990      0.000    -216.991    -103.548
유형[T.C]     -180.1103     40.883     -4.406      0.000    -266.778     -93.443
광고비            1.0095      0.652      1.548      0.141      -0.373       2.392
==============================================================================
Omnibus:                        9.726   Durbin-Watson:                   1.496
Prob(Omnibus):                  0.008   Jarque-Bera (JB):               16.186
Skew:                          -0.147   Prob(JB):                     0.000306
Kurtosis:                       7.397   Cond. No.                         382.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

""" 로지스틱 회귀 분석 """
from statsmodels.formula.api import logit
model = logit('Survived ~ C(Gender) + SibSp + Parch + Fare', data=df).fit()


""" odds """
"""
odds 비 : 두 odds의 비율
- 로지스틱 회귀에서는 오즈비를 사용해 독립변수가 한 단위 변할 때, 종속 변수의 오즈가 어떻게 변하는지 나타낸다.
    - 오즈비 > 1 : 독립변수의 증가가 종속변수의 확률을 증가
    - 오즈비 < 1 ; 독립변수의 증가가 종속변수의 확률을 감소
- odds 비를 구하는 방법
    > 특정 param의 계수가 a라면, 자연상수 e의 a제곱 : np.exp(a)
"""


""" 3. 분산분석 """
"""
[분산분석 (ANOVA)]
-> 3개 이상의 집단의 평균 차이를 검정
    - 일원 분산 분석 (one-way ANOVA) : 하나의 요인
    - 이원 분산 분석 (two-way ANOVA) : 2개의 요인
-> 기본 가정
    - 독립성
    - 정규성 (shapiro 검정 선행되어야 함)
    - 등분산성 (levene 검정 선행되어야 함)

[일원 분산 분석]
    - 3개 이상의 집단
    - 하나의 요인
    - 평균의 차이가 통계적으로 유의한가

- 귀무가설 : 모든 집단의 평균은 같다
- 대립가설 : 적어도 한 집단은 평균이 다르다

    {kruskal 검정 : 비모수 검정, 정규성을 만족하지 않을 때}

[이원 분산 분석]
    - 3개 이상의 집단
    - 2개의 요인
    - 평균의 차이가 통계적으로 유의한가

{요인 1}
- 귀무가설 : 요인1의 영향에도 모든 집단의 평균은 같다
- 대립가설 : 요인1의 영향으로 적어도 2개 집단의 평균은 다르다

{요인 2}
- 귀무가설 : 요인2의 영향에도 모든 집단의 평균은 같다
- 대립가설 : 요인2의 영향으로 적어도 2개 집단의 평균은 다르다

{요인 1과 2의 상호작용}
- 귀무가설 : 요인1과 요인2의 상호작용에도 모든 집단의 평균은 같다
- 대립가설 : 요인1과 요인2의 상호작용으로 적어도 2개 집단의 평균은 다르다

[사후 검증]
- tukey HSD & bonferroni : 어떤 집단간의 차이로 평균이 달라지는지 확인
"""

""" 일원 분산 분석 """
df = pd.DataFrame({
    'A': [3.5, 4.3, 3.8, 3.6, 4.1, 3.2, 3.9, 4.4, 3.5, 3.3],
    'B': [3.9, 4.4, 4.1, 4.2, 4.5, 3.8, 4.2, 3.9, 4.4, 4.3],
    'C': [3.2, 3.7, 3.6, 3.9, 4.3, 4.1, 3.8, 3.5, 4.4, 4.0],
    'D': [3.8, 3.4, 3.1, 3.5, 3.6, 3.9, 3.2, 3.7, 3.3, 3.4]
})

# 정규성 검정과 등분산 검정이 선행되어야 한다
# Shapiro-Wilk(샤피로-윌크) 정규성 검정
from scipy import stats
print(stats.shapiro(df['A']))
# ShapiroResult(statistic=0.949882447719574, pvalue=0.667110025882721)      > p-value > 0.05 : 정규성
print(stats.shapiro(df['B']))
# ShapiroResult(statistic=0.934644877910614, pvalue=0.49509894847869873)    > p-value > 0.05 : 정규성
print(stats.shapiro(df['C']))
# ShapiroResult(statistic=0.9871343374252319, pvalue=0.9919547438621521)    > p-value > 0.05 : 정규성
print(stats.shapiro(df['D']))
# ShapiroResult(statistic=0.9752339720726013, pvalue=0.9346861243247986)    > p-value > 0.05 : 정규성

# Levene(레빈) 등분산 검정
from scipy import stats
print(stats.levene(df['A'], df['B'], df['C'], df['D']))
# LeveneResult(statistic=1.5433829973707245, pvalue=0.22000894224209636)    > p-value > 0.05 : 등분산

# scipy 활용 (일원 분산만 가능)
from scipy import stats
stats.f_oneway(df['A'], df['B'], df['C'], df['D'])
# F_onewayResult(statistic=7.2969837587007, pvalue=0.0006053225519892207)

# statsmodel 활용
# 현재의 table 형태로는 불가 > table melt를 통해 나눠줌
# melt 함수를 통해 하기 형태로 재구조화
df_melt = df.melt()
"""
	variable	value
0	A	3.5
1	A	4.3
2	A	3.8
3	A	3.6
4	A	4.1
"""

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('value ~ variable', data=df_melt).fit()
print(anova_lm(model))
"""
           df   sum_sq  mean_sq         F    PR(>F)
variable   3.0  2.35875  0.78625  7.296984  0.000605
Residual  36.0  3.87900  0.10775       NaN       NaN
"""

""" 사후 검증 """
# 목적 : 어떤 그룹들 사이에서 통계적으로 차이가 발생하는지 확인하기 위함
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# tukey 검증
# 종속변수, 독립변수, 유의수준
tukey_result1 = pairwise_tukeyhsd(df_melt['value'], df_melt['variable'], alpha=0.05)
print(tukey_result1.summary())
# p-value < 0.05 : 대립가설 채택, 두 그룹 간에 유의한 차이가 있다, reject=True
"""
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
     A      B     0.41 0.0397  0.0146  0.8054   True
     A      C     0.09 0.9273 -0.3054  0.4854  False
     A      D    -0.27 0.2722 -0.6654  0.1254  False
     B      C    -0.32 0.1483 -0.7154  0.0754  False
     B      D    -0.68 0.0003 -1.0754 -0.2846   True
     C      D    -0.36 0.0852 -0.7554  0.0354  False
----------------------------------------------------
Test Multiple Comparison ttest_ind 
"""

# bonferroni 검증
mc = MultiComparison(df_melt['value'], df_melt['variable'])
bon_result = mc.allpairtest(stats.ttest_ind, method='bonf')
print(bon_result[0])
# p-value < 0.05 : 대립가설 채택, 두 집단 간 유의한 차이가 있다. reject=True
"""
FWER=0.05 method=bonf
alphacSidak=0.01, alphacBonf=0.008
=============================================
group1 group2   stat   pval  pval_corr reject
---------------------------------------------
     A      B -2.7199  0.014    0.0843  False
     A      C  -0.515 0.6128       1.0  False
     A      D  1.7538 0.0965    0.5788  False
     B      C  2.2975 0.0338    0.2028  False
     B      D  6.0686    0.0    0.0001   True
     C      D  2.5219 0.0213    0.1279  False
---------------------------------------------
"""

""" 크루스칼-왈리스 검정 (비모수 검정) : 정규성을 만족하지 않을 때 """
print(stats.kruskal(df['A']. df['B'], df['C'], df['D']))


""" 이원 분산 분석 """
# 영향을 주는 요인이 2개
# 각 요인은 물론, 요인 간 상호작용이 미치는 영향도 분석해야 함
# 상호작용을 포함하는 방법은
# 변수:변수 을 추가하는 방법으로 진행하거나
# 변수*변수 의 방법으로 진행 (+ 대신)

# ols 함수는 기본적으로 범주형 데이터를 자동으로 전환해줌
# 그러나 숫자로 범주형을 나타낸 경우가 있으면, 처리해줘야 함
# ols 학습할 때, C(변수명) 방식으로 해당 변수를 범주형 데이터로 전환한다

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('토마토수 ~ C(종자) + C(비료) + C(종자):C(비료)', data=df).fit()
# 결과 해석
# 종자 p-value < 0.05 : 대립가설 채택, 토마토수에 영향을 준다
# 비료 p-value < 0.05 : 대립가설 채택, 토마토수에 영향을 준다
# 종자와 비료 상호작용 p-value > 0.05 : 귀무가설 채택, 토마토수에 영향을 주지 않는다
"""
                df       sum_sq      mean_sq          F        PR(>F)
C(종자)          3.0  4801.000000  1600.333333  18.757977  7.254117e-10
C(비료)          2.0  1140.316667   570.158333   6.682993  1.835039e-03
C(종자):C(비료)    6.0   725.350000   120.891667   1.417007  2.146636e-01
Residual     108.0  9214.000000    85.314815        NaN           NaN
"""


""" 포아송 분포 """
"""
stats.poisson.pmf(k, mu)는 Poisson 분포에서 확률 질량 함수(probability mass function, PMF)를 
계산하는 SciPy의 메소드입니다. 
여기서 k는 특정 이벤트가 발생한 횟수를 나타내며, mu는 Poisson 분포의 평균 발생 횟수입니다.

그러면 stats.poisson.pmf(5, 3)은 평균 발생 횟수가 3인 Poisson 분포에서 
특정 이벤트가 5번 발생할 확률을 계산합니다. 즉, 이는 k=5인 경우의 Poisson 분포에서의 확률을 나타냅니다.
"""
from scipy.stats import poisson

# 평균 발생 횟수 (하루에 잡지를 구매하는 고객 수)
lambda_ = 3

# 하루에 정확히 5명의 고객이 잡지를 구매할 확률
print(poisson.pmf(5, lambda_))