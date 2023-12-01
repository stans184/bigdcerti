import pandas as pd
import numpy as np

# 모든 컬럼을 보여주는 pandas option
pd.set_option('display.max_columns', None)
# [Tip] 지수표현식(과학적표기법) 사용 X
pd.set_option('display.float_format', '{:5f}'.format)

# [Tip] set 타입으로 변경하면 비교 가능함
a = set(X_train['주구매상품'].unique())
b = set(X_test['주구매상품'].unique())
print(a-b)
print(a.difference(b))

# DataFrame으로 만들기
data = {
    "메뉴": ['아메리카노', '카페라떼', '카페모카', '바닐라라떼', '녹차', '초코라떼', '바닐라콜드브루'],
    "가격": [4100, 4600, 4600, 5100, 4100, 5000, 5100],
    "할인율": [0.5, 0.1, 0.2, 0.3, 0, 0, 0],
    "칼로리": [10, 180, 420, 320, 20, 500, 400],
}
data = pd.DataFrame(data)

# index 없이 저장하기
data.to_csv('data.csv', index=False)

# 데이터 불러오기
pd.read_csv('data.csv')
# 데이터 샘플 확인: 앞에서 부터 n개 (기본 5개)
data.head()
# 데이터 샘플 확인: 뒤에서 부터 n개 (기본 5개)
data.tail(3)

# 시리즈 만들기 (문자열)
menu = pd.Series(['iced americano', 'iced latte', 'espresso'])
# 시리즈 만들기(정수형)
price = pd.Series([2000, 3000, 2500])
# 데이터 프레임 만들기 pd.DataFrame({"컬럼명":데이터})
cafe = pd.DataFrame({
    "menu": menu,
    "price": price
})
# 대괄호를 2번 하면 DataFrame 형태로 선택됨
cafe[['price']]

# 데이터 프레임 크기 (행, 컬럼)
cafe.shape
# 컬럼 형태(type)
cafe.info()
# 기초 통계
cafe.describe()
# 기초 통계 (object)
cafe.describe(include='O')
# 범주형 컬럼 통계값 확인 (train)
cafe.describe(include='object')
# 상관관계
# 상관관계는 1로 갈수록 양의 상관관계
# -1로 갈수록 음의 상관관계가 강해지는 것
# 따라서 상관관계의 비교는 상관계수의 절대값 크기로 비교해야 한다
cafe.corr()

# 항목 종류 수
cafe.nunique()
# 항목 종류
print(cafe['menu'].unique())
print(cafe['price'].unique())
# 항목별 개수
print(cafe['menu'].value_counts())
# 각 항목의 비율
print(df['test'].value_counts(normalize=True))

# 데이터 프레임 만들기 (할인율과 칼로리 -> 문자열)
data = {
    "메뉴":['아메리카노', '카페라떼', '카페모카', '바닐라콜드브루'],
    "가격":[4100, 4600, 4600, 5100],
    "할인율":['0.5', '0.1', '0.2', '0.3'],
    "칼로리":[10,180,420,320],
}
cafemenu = pd.DataFrame(data)
cafemenu.info()
# 데이터프레임 선택 (가격) => 대괄호를 2번 하면 dataframe으로 선택됨
cafemenu[['price']]
# 자료형 변환 / astype /  object -> float
cafemenu['할인율'] = cafemenu['할인율'].astype(float)
# 할인가 컬럼 추가
cafemenu['할인가'] = cafemenu['가격']*(1-cafemenu['할인율'])
# 결측값으로 추가, 원두 컬럼을 만들고 결측값(NaN)으로 대입
cafemenu['원두'] = np.nan

""" 데이터 삭제 """
# axis=1:열방향(컬럼) / axis=0:행방향
cafemenu = cafemenu.drop('new', axis=1)
# 행 삭제
cafemenu = cafemenu.drop(3, axis=0)
# 결측치가 있는 데이터(행) 전체 삭제 및 확인 dropna() #기본값 axis=0
# dropna : null 값이 들어 있는 axis=0, 행을 아예 삭제시킴
cafemenu.dropna()
# 특정컬럼에 결측치가 있으면 데이터(행) 삭제 subset=['native.country']
# 특정 row만 삭제할 때는 axis=0
df = X_train.dropna(subset=['native.country', 'workclass'])
# 결측치가 있는 컬럼 삭제 dropna(axis=1)
# 잘 없음, 컬럼은 결국 feature 인데, 이걸 다 날려버리기 떄문
# 작업형 1번 문제에서는 결측치가 있는 column을 다 날리라는 식으로 나올 수도 있음
# 지문을 잘 읽어보는 것이 중요
df = X_train.dropna(axis=1)
# 결측치가 많은 특정 컬럼 삭제 drop(['workclass'], axis=1)
df = X_train.drop(['workclass'], axis=1)
# inplace True > 바로 저장됨
train.drop('Attrition_Flag', axis=1, inplace=True)


""" 인덱싱, 슬라이싱 """
# loc : 인덱스 명(범위), 컬럼 명(범위), 포함됨
# iloc : 인덱스 번호(범위), 컬럼 번호(범위), 프로그래밍 인덱스, 마지막 미포함
cafemenu.loc[0:2, '메뉴':'할인율']
cafemenu.iloc[0:3, 0:3]


""" sorting """
# 인덱스 기준 (기본값 ascending=True)
cafemenu.sort_index(ascending=False)
# 값 기준 (기본값 ascending=True)
cafemenu.sort_values('가격', ascending=False)
# 가격과 메뉴 기준 정렬
cafemenu = cafemenu.sort_values(['가격', '메뉴'], ascending=[False, True])


""" 조건 필터 """
# 2개 이상 일때 (AND)
# 할인율 >= 0.2
# 칼로리 < 400
cond1 = cafemenu['할인율'] >= 0.2
cond2 = cafemenu['칼로리'] < 400
cafemenu[cond1 & cond2]
# 2개 이상 일때 (OR)
# 할인율 >= 0.2
# 칼로리 < 400
cond1 = cafemenu['할인율'] >= 0.2
cond2 = cafemenu['칼로리'] < 400
cafemenu[cond1 | cond2]
# 어떤 조건 이외의 값들
cafemenu[~cond1]


""" 조건 변경 """
# 'Events'가 1일 때, 'Sales'를 80%로 저장
df.loc[df['Events'] == 1, 'Sales'] = df.loc[df['Events'] == 1, 'Sales']*0.8
# apply를 사용한 조건 변경
def event_sales(x):
    if x['Events'] == 1:
        x['Sales2'] = x['Sales']*0.8
    else:
        x['Sales2'] = x['Sales']
    return x

df = df.apply(lambda x: event_sales(x), axis=1) #1일 경우 row, 0일 경우 컬럼


""" 누적합 """
# 예시 데이터프레임 생성
df = pd.DataFrame({'A': [1, 2, 3, 4]})
# 'A' 열의 누적 합 계산
cumulative_sum = df['A'].cumsum()
# 결과 확인
print(cumulative_sum)


""" 결측치 """
# 결측치는 최빈값과 차이가 크면 최빈값으로 값이 비슷하면 별도의 값으로 대체함
# 컬럼별 결측치 확인
cafemenu.isnull().sum()
# 결측값 채우기
# 원두-> 코스타리카로 채우기
cafemenu['원두'] = cafemenu['원두'].fillna('코스타리카')
# 최빈값으로 채우기
X_train['workclass'] = X_train['workclass'].fillna(X_train['workclass'].mode())
# 앞, 뒤의 값으로 결측치 채우기
df.fillna(method='bfill')
df.fillna(method='ffill')


""" 여 존슨, box-cox 변환값 """
from sklearn.preprocessing import power_transform
data = [[11, 12], [23, 22], [34, 35]]
print(power_transform(data)) # method 디폴트 값은 여-존슨’yeo-johnson’
print(power_transform(data, method='box-cox'))


""" 값 변경 """
# 문자 변경 : 아메리카노 -> 룽고, 녹차 -> 그린티
df = cafemenu.replace('아메리카노', '롱고').replace('녹차', '그린티')
# 데이터의 일부를 변경하고 싶다면, str을 이용해서 변경해야 함
df = cafemenu.str.replace('블랙', '화이트')
# 문자 변경2 : dict을 이용한 변경
d = {'롱고' : '아메리카노', '그린티' : '녹차'}
cafemenu = cafemenu.replace(d)
# map을 활용한 변경
dict_list = {np.nan : 0, 'silver' : 1, 'gold' : 2, 'vip' : 3}
df = df.map(dict_list)


""" * pandas 내장함수 * """
# 데이터 불러오기
df = pd.read_csv('data.csv')
# 카운트 (컬럼) #기본값 axis=0
df.count()
# 카운트 (행)
# nan은 제외하고 카운팅 됨
df.count(1)
# [Tip] 데이터 수 len, shape
print(len(df))
df.shape # 데이터 수, 컬럼 수
# 최대값
df['가격'].max()
# 최소값
df['가격'].min()
# 평균
df['가격'].mean()
# 중앙값
df['가격'].median()
# 합계
df['가격'].sum()
# 표준편차
df['가격'].std()
# 분산
df['가격'].var()
# 왜도 skewness
df['가격'].skew()
# 첨도 kurtosis
df['가격'].kurt()
# 백분위수
df.describe()
# 하위 25% 값
df['가격'].quantile(0.25)
# 하위 25% 데이터
cond = df['가격'].quantile(0.25) > df['가격']
df[cond]
# 상위 25% 값
df['가격'].quantile(0.75)
# 최빈값
df['원두'].mode()[0]
# 혹은, string으로 취급해서 slicing 하기
df['날짜'].str[:4]
# 트랜스포스, 행과 열을 바꿔버림
df = df.T


""" 행별 총계를 저장하기 """
# 2번쨰 열 이후를 모두 선택해서
# 합계를 저장
# df의 예시를 생각해보면 쉬움
# ['1학년 수', '2학년 수', '3학년 수'] 의 컬럼이 있는 데이터 프레임일 때
# 전교생을 알고 싶다면 아래와 같이 행별 총계를 더해서 새로운 column으로 만들 수 있음
# sum(axis=0) 은 default 값, 즉 해당 column 의 총합을 구하는 것
df['전교생수'] = df.iloc[:, 2:].sum(axis=1)


""" 가장 큰 값의 index 가져오기 """
# 특정 column에서 최대값의 index를 가져올 때 편함
df['교사당학생'].idxmax()


""" 외부 함수 적용 """
# apply 예시
def cal(x):
    return "No" if x>=300 else "Yes"
df['칼로리'].apply(cal)
# apply 적용해서 새로운 컬럼 생성 (칼로리 컬럼 활용)
df['CanIEat'] = df['칼로리'].apply(cal)


""" grouping """
# 원두 기준, 평균
df.groupby('원두').mean()
# 원두와 할인율 기준, 평균
df.groupby(['원두', '할인율']).mean()
# 원두와 할인율 기준, 가격 평균
df.groupby(['원두', '할인율'])['가격'].mean()
# 원두와 할인율 기준, 가격 평균 -> 데이터 프레임 형태
df.groupby(['원두', '할인율'])[['가격']].mean()
# pd.DataFrame(df.groupby(['원두', '할인율'])['가격'].mean())
# 1개 인덱스 형태로 리셋
df.groupby(['원두', '할인율']).mean().reset_index()


""" 데이터 합치기 & 분리하기 """
# X_train y_train 합치는 것 예시
# axis 설정 필수, 0은 행을 기준으로(밑으로), 1은 열을 기준으로(옆으로)
# 합치고자 하는 데이터를 []로 감싸서 진행 -> dataframe끼리만 가능
df = pd.concat([X_train, y_train['income']], axis=1)
# train 분리 예시
X_tr = train.iloc[:, :-1].copy()
y_tr = train.iloc[:, [0,-1]].copy()
##
# 리스트에서 리스트 빼기
cols = list(filter(lambda x : x not in list2, list1))


""" IQR """
# IQR로 확인
cols = ['age','fnlwgt','education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
for col in cols:
    Q1 = X_train[col].quantile(.25)
    Q3 = X_train[col].quantile(.75)
    IQR = Q3 - Q1
    min_iqr = Q1-1.5*IQR
    max_iqr = Q3+1.5*IQR
    cnt=sum((X_train[col] < min_iqr) | (X_train[col] > max_iqr))
    print(f'{col}의 이상치:{cnt}개 입니다.')
    

""" dtype을 활용한 수치형 / 범주형 데이터 분리 """
# 수치형 컬럼과 범주형 컬럼 데이터 나누기
# select_data type
# n_train = X_train.select_dtypes(exclude='object').copy()
# n_test = X_test.select_dtypes(exclude='object').copy()
# c_train = X_train.select_dtypes(include='object').copy()
# c_test = X_test.select_dtypes(include='object').copy()
# 데이터를 매번 새롭게 불러오기 위해 함수로 제작 함
def get_nc_data():
    n_train = X_train.select_dtypes(exclude='object').copy()
    n_test = X_test.select_dtypes(exclude='object').copy()
    c_train = X_train.select_dtypes(include='object').copy()
    c_test = X_test.select_dtypes(include='object').copy()
    return n_train, n_test, c_train, c_test

n_train, n_test, c_train, c_test = get_nc_data() # 데이터 새로 불러오기


""" datetime """
df['A'] = pd.to_datetime(df['A'])
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y년 %m월')
df['year'] = df['A'].dt.year
df['month'] = df['A'].dt.month
df['day'] = df['A'].dt.day
df['hour'] = df['DateTime4'].dt.hour
df['minute'] = df['DateTime4'].dt.minute
df['second'] = df['DateTime4'].dt.second

# Date2 (year 4자리: %Y) ex) 2024:02:18
df['Date2'] = pd.to_datetime(df['Date2'], format='%Y:%m:%d')
# Date3 (year 2자리: %y) ex) 24/02/17
df['Date3'] = pd.to_datetime(df['Date3'], format='%y/%m/%d')
# 18-Feb-2024 -> format 지정 안해줘도 됨
# ex) 24-02-17 11:45:30
df['DateTime1'] = pd.to_datetime(df['DateTime1'], format='%y-%m-%d %H:%M:%S')

# [참고] 기간 to_period()
print(df['DateTime4'].dt.to_period('Y')) # 2024
print(df['DateTime4'].dt.to_period('Q')) # 2024Q1
print(df['DateTime4'].dt.to_period('M')) # 2024-02
print(df['DateTime4'].dt.to_period('D')) # 2024-02-17
print(df['DateTime4'].dt.to_period('H')) # 2024-02-17 11:00

# 주말과 주중을 구분할 수 있는 함수
# 둘 다 같은 값을 출력함
# 요일 dayofweek 0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
df['DateTime4'].dt.dayofweek
df['weekday'] = df['A'].dt.weekday

# 몇번째 주인지 출력
print(df['date'].dt.isocalendar().week)

# timedelta : 특정 시간 이후, 이전
# 시간 +/- (3주, 3일, 3시간, 3분, 3초 이전과 이후)
delta = pd.Timedelta(weeks=3, days=3, hours=3, minutes=3, seconds=3)
print(df['DateTime4'] - delta)

# timedelta64
# datetime 자료형의 차이
# total_seconds() : 총 기간을 초로 환산
# days : 몇 일 간격인지만 보여줌
# seconds : 시간 간격의 단위만 초로 바꿔서 보여줌
print(df['diff'])
print(df['diff'].dt.total_seconds())
print(df['diff'].dt.days)
print(df['diff'].dt.seconds)

# 일종의 groupby 와 유사한 개념으로 resample이 있음
# 이거는 시계열 데이터를 대상으로
# resample('D') 하면 날짜별로
# resample('W') 하면 주별로
# resample을 사용하고자 한다면, index가 date 형태로 잡혀있어야 함 (index_col=0)
df.resample('W').sum()


""" 구간분할 """
# df['age']를 q=3에 맞춰서 나누고, labels에 맞게 지정한 값을 저장
# 데이터를 분위수를 기준으로 q값만큼 분할해서 저장
df['range'] = pd.qcut(df['age'], q=3, labels=['group1','group2','group3'])


""" 중복제거 """
# subset : 중복 제거를 진행할 특정 column 선정
# keep : 중복 제거 방향
#   - first : 처음에 나온 행을 남김
#   - last : 마지막에 나온 행을 남김
#   - False : 중복된 모든 행 삭제
df.drop_duplicates(subset=['age'], keep='first', inplace=True)


""" 데이터 한칸씩 밀기 / 당기기 """
df.shift(1)


""" 문자열을 건드리고 싶다면 .str """
df['menu'].str.contains('라떼')
df['menu'].str.split(" ", '')
df['menu'].str.replace('a', 'b')
df['menu'].str[0:4]


""" dataframe 녹이기 """
# melt : wide format 에서 long format으로 변경
# id_vars : 변환 후에도 유지할 변수들의 열 이름
# value_vars : long format으로 변환할 변수 이름
df = pd.melt(df, id_vars=['Name'], value_vars=['수학', '영어'])
"""
0 민수 수학 54
1 철수 수학 87
2 영희 영어 43
3 민수 영어 86
4 영희 수학 65
"""