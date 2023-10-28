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
1. 샤피로-윌크 검정 (정규성 확인) : shapiro(diff) diff = A-B
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
# 'two-sided' : 크거나, 작거나 양측 검정
# 'greater' or 'less' : 큰 값, 작은 값 단측 검정
from scipy import stats
print(stats.ttest_1samp(df['무게'], 120, alternative='two-sided'))
# TtestResult(statistic=2.153709967150663, pvalue=0.03970987897788578, df=29)

""" 샤피로 윌크 검정 : 정규성 만족 여부 확인 """
# 귀무가설 (H0): 주어진 데이터 샘플은 정규 분포를 따른다.
# 대립가설(H1): 주어진 데이터 샘플은 정규 분포를 따르지 않는다.
# Shapiro-Wilk(샤피로-윌크) 정규성 검정
# p-value < 0.05 : 대립가설 채택, 정규분포를 따르지 않는다
# p-value > 0.05 : 귀무가설 채택, 정규분포를 따른다
from scipy import stats
stats.shapiro(df['무게'])

""" 윌콕슨 검정 (비모수 검정)) : 정규성을 따르지 않는 데이터에 대해 """
# 귀무가설 (H0): μ = μ0, "합격 원두(dark)" 상품의 평균 무게는 120g이다.
# 대립가설(H1): μ < μ0, "합격 원두(dark)" 상품의 평균 무게는 120g 보다 작다
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
from scipy import stats
stats.shapiro(A)
stats.shapiro(B)

""" Levene 검정 : 등분산 검정 """
# 두 집단이 모두 정규성을 만족할 때
# 동일한 분산을 가지고 있는지 검정
from scipy import stats
stats.levene(A, B)

""" 독립 표본 검정 진행 """
# 최소한 두 집단 모두 정규성을 만족할 때 진행되어야 함
# 등분산 여부도 확인해서, 그에 따른 옵션 설정 필요
# 분산이 같으면 True, 다르면 False
from scipy import stats
stats.ttest_ind(A, B, equal_var=True)

""" Mann-whitneyu 검정 : 정규성을 만족하지 않을 때 """
# 두 집단 중 하나라도 정규성을 만족하지 않을 때
from scipy import stats
stats.mannwhitneyu(A, B, alternative='less')