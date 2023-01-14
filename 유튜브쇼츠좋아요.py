pip install selenium

from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
 
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import random
from selenium.webdriver.common.by import By

driver = webdriver.Chrome(ChromeDriverManager().install())

movie = pd.read_csv('유튜브쇼츠_1025.csv')

list = []
movie_list = []
like_list = []
   
for i in movie.loc[:,'주소'] :
    url = i
    driver.get(url)
    time.sleep(2)
    
    try:           
        soup = bs(driver.page_source, 'html.parser')
        like = driver.find_element(By.CSS_SELECTOR,'.style-scope ytd-toggle-button-renderer').text.split('\n')
        movie_list.append(i)
        like_list.append(like)

    except:    
        i


youtubeDic = {
      '영화제목' : movie_list,
      '좋아요수': like_list}
youtubeDf = pd.DataFrame(youtubeDic)
  
    
youtubeDf.to_csv( '유튜브쇼츠좋아요수.csv', encoding='', index=False)
    



movie1 = pd.read_csv('유튜브쇼츠좋아요수.csv')
