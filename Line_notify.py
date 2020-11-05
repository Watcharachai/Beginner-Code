#!/usr/bin/env python
# coding: utf-8

# # Import Packages and Produce Functions

# In[1]:


import pandas as pd
import time
import requests
import numpy as np
import schedule
import pip
import stop_words
import pythainlp
from pythainlp import word_tokenize
#from pythainlp.corpus import stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
import tweepy
import random
import re
import string
import stop_words
import nltk
from pythainlp.corpus import thai_stopwords


# In[2]:


import datetime
from datetime import date


# # Pull Current Data

# In[3]:


#Define important words
bag_words = ['อุบัติเหตุ','ซ่อม','ฝนตก','วิภาวดี','แคราย','สะพานพระนั่งเกล้า','รัตนาธิเบศ','พงษ์เพชร','แยกเกษตร','งามวงศ์วาน','บางเขน']
#Binary class of trafic condition
condition = ('good', 'bad') #labels
#Negatively important words
neg_word = ['อุบัติเหตุ','ซ่อม','ฝนตก','จอดเสีย','รถชน','เคลื่อนย้าย','ติดขัด','สะสม','มาก','ท้ายแถว']
place_word = ['วิภาวดี',
 'แคราย',
 'สะพานพระนั่งเกล้า',
 'รัตนาธิเบศ',
 'พงษ์เพชร',
 'เกษตร',
 'งามวงศ์วาน',
 'บางเขน']
place_word_ref = place_word


# In[7]:


consumer_key = "sOnn3bjnHhG6zKKzqbZpW6Ccs"
consumer_secret = "f5TO0onnROxiE04QbLdyXMBAt2YCc8LIMmtqQTbGQk3yuCYUDk"
access_token = "1320225693939101696-zko7CVgLHvLMFVpzW4wWsJKnIITtss"
access_token_secret = "OYAfv4Zz0vEjpAyfIs5PMlkUTB6X4l2TZGb6c87LWGmWM"
#I suggest to regen keys and tokens everytime coding

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
api = tweepy.API(auth)

a = api.get_status(912886007451676672, tweet_mode='extended')


# scrape data from #query

# In[4]:


#Create random to def
def sampling_func(items,k):
    #create random samples for printing news (consequence)
    sampling_neg = random.sample(items, k)
    #create random samples for printing news (place)
    sampling_place = random.sample(place_word_ref,k=1)
    #concat neg & place
    sampling_neg.extend(sampling_place)
    sampling_neg.append('#รถติด') #necessary to add a vital keyword
    return sampling_neg


# In[5]:


sampling=sampling_func(neg_word,1)


# In[8]:


#scrape data from twitter query
query = sampling_func(neg_word,1)
count = 100
try:
    tweets_q =tweepy.Cursor(api.search, q= query,full_text = True,result_type = 'recent'
    ,until_date = date.today()).items(count)
    tweets_list_q = [[obj.created_at, obj.id, obj.text] for obj in tweets_q]
    tweets_df_q = pd.DataFrame(tweets_list_q,columns=('date_time','id', 'tweet'))
    while True:
        if len(tweets_df_q.tweet) == 0:
            query = sampling_func(neg_word,1)
            tweets_q =tweepy.Cursor(api.search, q= query,full_text = True,result_type = 'recent'
            ,until_date = date.today()).items(count)
            tweets_list_q = [[obj.created_at, obj.id, obj.text] for obj in tweets_q]
            tweets_df_q = pd.DataFrame(tweets_list_q,columns=('date_time','id', 'tweet'))
        else:
            break
except BaseException as e:
    print('failed on_status,',str(e))
    time.sleep(3)


# In[9]:


tweets_df_q = tweets_df_q.drop(['date_time','id'],axis=1)
#tweets_df_q=tweets_df_q[tweets_df_q['tweet'].str.contains("Trump")] #Change Everytime!!!


# In[10]:


tweets_df_q.drop_duplicates(keep="first", inplace=True) #Drop duplicated items


# # Pre processing Data

# In[14]:


def clean_msg(msg): #Clear all signs
    
    # delete text in <>
    msg = re.sub(r'<.*?>','', msg)
    
    # delete hashtag
    msg = re.sub(r'#','',msg)
    
    # delete punctuation
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    # delete separator i.e. \n \t
    msg = ' '.join(msg.split())
    
    return msg


# In[15]:


nltk.download('words') #pull thai word(Bags)
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()


# In[16]:


def split_word(text):       
    
    tokens = word_tokenize(text,engine='newmm')
    
    # Remove stop words
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]
    
    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]
    
    # Thai
    tokens_temp=[]
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    
    tokens = tokens_temp
    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]
    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]
    return tokens


# # Choose only one

# In[ ]:


#Set New dataframe by user
df = tweets_df


# In[17]:


#Set New dataframe by query
df = tweets_df_q


# In[18]:


list_clean_df = [clean_msg(i) for i in df.tweet] #clean msg
list_token_df = [split_word(text) for text in list_clean_df] #split words


# In[19]:


#Add token to df
df['token'] = list_token_df


# In[20]:


label_total = []
for i in range(len(df.token)):
    label = [txt for txt in df.token.iloc[i] if txt in neg_word]
    label = list(set(label))
    label_total.append(label)
    print(i)
    print(label)


# In[21]:


place_total = []
for i in range(len(df.token)):
    place = [txt for txt in df.token.iloc[i] if txt in place_word]
    place = list(set(place))
    place_total.append(place)
    print(place)
    print(place)
    
#Not accurate as the split word is not perfect
#Intial solution ==>> Add more place_word

place_word.extend(['รัตนา','เพชร','เกล้า','เขน', 'บาง', 'วัฒนะ','แค','ราย', 'งาม', 'วาน'])
#concat full word
for aa in place_total:
    for i in range(len(aa)):
        if aa[i] == 'เขน'or aa[i] == 'บาง':
             aa[i] = 'บางเขน'
        elif aa[i] == 'แค'or aa[i]=='ราย':
            aa[i]='แคราย'
        elif aa[i]=='เกล้า':
            aa[i]='สะพานพระนั่งเกล้า'
        elif aa[i]=='รัตนา'or aa[i]=='ธิเบศ':
            aa[i]= 'รัตนาธิเบศ'
        elif aa[i]=='พงศ์'or aa[i]=='เพชร':
            aa[i]='พงศ์เพชร'
        elif aa[i]=='งาม'or aa[i]=='วงศ์'or aa[i]=='วาน':
            aa[i]='งามวงศ์วาน'
        elif aa[i]=="แจ้ง" or aa[i]=='วัฒนะ':
            aa[i]='แจ้งวัฒนะ'
        else:
            pass


# In[22]:


df['condition'] = label_total #gern condition to df
df['place'] = place_total #gern place to df


# In[23]:


df


# # Notify

# In[24]:


def Lineconfig(command):
	url = 'https://notify-api.line.me/api/notify'
	token = 'tUSCy31qagrhdtM5SsNUf5U8s4glqLEo21dELnG2D1n' ## EDIT
	header = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
	return requests.post(url, headers=header, data = command)

def sendtext(message):
	# send plain text to line
	command = {'message':message}
	return Lineconfig(command)

def sendcon(condition):
	# send condition
	command = {'message':condition}
	return Lineconfig(command)

def sendplace(place):
	# send place
	command = {'message':place}
	return Lineconfig(command)

def sendnews(news):
	# send news
	command = {'message':news}
	return Lineconfig(command)


# In[25]:


#define time
i = date.today().weekday()
if i == 1:
    time = 74
elif i == 2:
    time = 69
elif i == 3:
    time = 59
elif i == 4:
    time = 57
elif i == 5:
    time = 54
else:
    time="Weekend"


# In[26]:


#Execute program
def execute_notice(time,condition, place):
    sendtext("Good Morning")
    sendtext("ใช้ระยะเวลาเดินทางทั้งหมด {} นาที".format(time))
    sendcon("มีปัญหาเกิดจาก {}".format(condition))
    sendplace("โปรดหลีกเลี้ยงเส้นทาง {}".format(place))
    
def execute_news(news):
    sendnews("สำหรับข่าวเพิ่มเติม {}".format(news))


# In[27]:


execute_notice(59,df.condition.iloc[0],df.place.iloc[0])

for i in range(0,2): # 2 latest news
    execute_news(news=df.tweet.iloc[i])


# In[ ]:




