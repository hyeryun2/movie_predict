pip install webdrivermanager
pip install webdriver_manager
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import random
from selenium.webdriver.common.by import By
import datetime 


driver = webdriver.Chrome(ChromeDriverManager().install())
from bs4 import BeautifulSoup

movie = pd.read_csv('예고편_1030.csv', encoding = 'CP949')


movie_list = []
like_list = []
   
for i in movie.loc[:,'주소'] :
    url = i
    driver.get(url)
    time.sleep(2)
    driver.find_element("xpath", '//*[@id="info-container"]').click()
    # driver.find_element(By.CSS_SELECTOR,'style-scope ytd-watch-metadata').click()
    
    
    try:           
        
        
        soup = bs(driver.page_source, 'html.parser')
        date_text = soup.find_all('span','style-scope yt-formatted-string bold')[-1].string
        
        movie_list.append(i)
        like_list.append(date_text)

    except:    
        url


youtubeDic = {
      '주소' : movie_list,
      '게시일': like_list}
youtubeDf = pd.DataFrame(youtubeDic)
  
    
youtubeDf.to_csv( '채널명태그.csv', encoding='', index=False)

 # like = [soup.find_all('a','yt-simple-endpoint style-scope yt-formatted-string')[n].string for n in range(0,len(channel))]