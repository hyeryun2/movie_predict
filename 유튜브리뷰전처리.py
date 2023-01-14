import pandas as pd

movie = pd.read_csv('50위영화별유튜브_1030.csv')

movie1 = movie.drop_duplicates(['영상제목','주소'],keep ='first')

movie1['조회좋아요수']= movie1['조회좋아요수'].name.replace('[좋아요]','0회')

movie1.to_csv( '50위영화별유튜브정제_1030.csv', encoding='', index=False)


movie1 = pd.read_csv('유튜브정제_1030.csv', encoding = 'CP949')

word.strip('회') for word in movie1['조회좋아요수']

# =============================================================================
# movie1 = movie1.applymap(lambda x : x if pd.isnull(x) else str(x).replace(',',''))  
# movie1 = movie1.applymap(lambda x : x if pd.isnull(x) else str(x).replace('회',''))
# movie1 = movie1.applymap(lambda x : x if pd.isnull(x) else str(x).replace('[\'','')) 
# movie1 = movie1.applymap(lambda x : x if pd.isnull(x) else str(x).replace('\']','')) 
# 
# movie2 = movie1.iloc[:,[2,-1]]
# movie2 = movie2.applymap(lambda x : x if pd.isnull(x) else str(x).replace('좋아요','0'))
# movie2 = movie2.applymap(lambda x : x if pd.isnull(x) else str(x).replace('만','000'))
# movie2 = movie2.applymap(lambda x : x if pd.isnull(x) else str(x).replace('천','00'))
# movie4 = movie1.iloc[:,[-2,-1]]
# 
# =============================================================================
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('.','')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace(',','')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('회','')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('[\'','')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('\']','')) 

movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('좋아요','0')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('만','000')) 
movie1['조회좋아요수'] = movie1['조회좋아요수'].map(lambda x : x.replace('천','00')) 

movie1

movie1.to_csv( '유튜브정제_1031.csv', encoding='', index=False)
movie1.sort_values(by='Unnamed: 0')

movie2 =  pd.read_csv('유튜브정제_1031.csv', encoding = 'CP949')
movie2['조회좋아요수'].replace('분',NA,regex=True)

mask = movie2['조회좋아요수'].isin(['분'])
movie3 = movie2[~mask]
