# =============================================================================
# 변수중요도 시각화 
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from sklearn.datasets import load_boston
import seaborn as sns

var_imp = pd.read_csv('변수중요도_real.csv')
var_imp = var_imp.sort_values('중요도', ascending = False)
var_imp = var_imp.iloc[:,1:]

df_drop = df[df.Stats != '+/-']
import plotly.graph_objects as go
d={"Stats" : var_imp["이름"] , "FI" : var_imp["중요도"]}
fig = px.bar_polar(var_imp, r="FI", theta="Stats",
                   color="Stats", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r,)
fig.show()

import plotly.graph_objects as go
fig = px.bar_polar(var_imp, r="이름", theta="중요도",
                    template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r,)
fig.show()

# df_fi = pd.DataFrame({'columns':X.columns, 'importances':feature_importance})
# df_fi = df_fi[df_fi['importances'] > 0] # importance가 0이상인 것만 
# df_fi = df_fi.sort_values(by=['importances'], ascending=False)

fig = plt.figure(figsize=(15,7))



palette = sns.color_palette('Set3')
sns.palplot(palette)


plt.rc('font', family='Malgun Gothic')   
colors = sns.color_palette('pastel',len('이름'))
a1 = var_imp.plot.bar(x='이름',y='중요도', color = colors)
a1.set_xticklabels(var_imp['이름'], rotation=80, fontsize=9.5)
plt.tight_layout()
plt.title("영화 관람객수 예측 변수 중요도")
plt.show()




ax = var_imp.barplot(var_imp['이름'], var_imp['중요도'])
ax.set_xticklabels(var_imp['이름'], rotation=80, fontsize=13)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt # 득점모델 변수 중요도
import seaborn as sns

ftr_importances_values = rf_run.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_train11.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]
 
plt.figure()
sns.barplot(x=var_imp['중요도'], y=var_imp['이름'])
plt.title("영화 관람객수 예측 변수 중요도")
plt.show()


# fig = plt.figure()
# fig = px.bar(var_imp, x='중요도', y='이름', template="plotly_dark")
# fig.show()

# =============================================================================
# 변수중요도_요인팀
# =============================================================================
tt = pd.read_csv('변수중요도_요인팀.csv')
plt.figure()
sns.barplot(x=tt['변수중요도'], y = tt['변수'])
plt.title("영화 흥행 요인 분석 변수 중요도")
plt.show()


# =============================================================================
# 변수중요도요인 vs 예측
# =============================================================================
tt = tt.iloc[:5,]
plt.figure()
sns.barplot(x=tt['변수'], y = tt['변수중요도'])
plt.title("영화 흥행 요인 분석 중요 변수")
plt.show()

tt = pd.read_csv('변수중요도_요인팀.csv')
tt = tt.iloc[:5,]
plt.figure()
sns.barplot(x=tt['변수'], y = tt['변수중요도'])
plt.title("영화 흥행 요인 분석 중요 변수")
plt.show()
colors = sns.color_palette('pastel')


var_imp = pd.read_csv('변수중요도_real.csv')
var_imp = var_imp.sort_values('중요도', ascending = False)
var_imp = var_imp.iloc[:5,]
var_imp.columns = ['NO', '변수', '변수중요도']


plt.figure()
sns.barplot(x=var_imp['변수'], y = var_imp['변수중요도'])
plt.title("영화 관람객수 예측 중요 변수")
plt.show()





plt.figure()
plt.scatter(tt['블로그리뷰수'], tt['관객수'])
plt.show()

plt.figure()
plt.scatter(tt['예고편조회수'], tt['관객수'])
plt.show()


import seaborn as sns
plt.rc('font', family='Malgun Gothic')   
sns.pairplot(tt, hue="관객수")


# =============================================================================
# 이상치 제거
# =============================================================================

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

data = tt.loc[:,['블로그리뷰수','예고편조회수','관객수']]

data_mean = np.mean(data)
data_std = np.std(data)
pdf = stats.norm.pdf(np.sort(data), data_mean, data_std)

plt.figure()
plt.plot(np.sort(data), pdf)

t1 = tt.loc[:,['블로그리뷰수','관객수']]
t2 = tt.loc[:,['예고편조회수','관객수']]

data_mean = np.mean(t1)
data_std = np.std(t1)
pdf = stats.norm.pdf(np.sort(t1), data_mean, data_std)

plt.figure()
plt.plot(np.sort(data), pdf)

s_ol_data = pd.Series(ol_data)
level_1q = s_ol_data.quantile(0.25)
level_3q = s_ol_data.quantile(0.75)
IQR = level_3q - level_1q
rev_range = 3  # 제거 범위 조절 변수
dff = s_ol_data[(s_ol_data <= level_3q + (rev_range * IQR)) & (s_ol_data >= level_1q - (rev_range * IQR))]
dff = dff.reset_index(drop=True)

# =============================================================================
# 다시
# =============================================================================
def outlier_iqr(data, column): 

    # lower, upper 글로벌 변수 선언하기     
    global lower, upper    
    
    # 4분위수 기준 지정하기     
    q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)          
    
    # IQR 계산하기     
    iqr = q75 - q25    
    
    # outlier cutoff 계산하기     
    cut_off = iqr * 1.5          
    
    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off     
    
    print('IQR은',iqr, '이다.')     
    print('lower bound 값은', lower, '이다.')     
    print('upper bound 값은', upper, '이다.')    
    
    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기     
    data1 = data[data[column] > upper]     
    data2 = data[data[column] < lower]    
    
    # 이상치 총 개수 구하기
    return print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')
    

outlier_iqr(t1,'블로그리뷰수')
data = t1[(t1['블로그리뷰수'] < upper) & (t1['블로그리뷰수'] > lower)]
len(data)

plt.figure()
plt.scatter(data['블로그리뷰수'], data['관객수'])
plt.show()


# =============================================================================
# 관람횟수
# =============================================================================
mm = pd.read_csv('관람횟수.csv')
