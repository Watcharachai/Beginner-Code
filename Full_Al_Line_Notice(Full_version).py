#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Watcharachai/Beginner-Code/blob/master/Full_Al_Line_Notice.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
#Negatively important words
neg_word = ['อุบัติเหตุ','ซ่อม','ฝนตก','จอดเสีย','รถชน','เคลื่อนย้าย','ติดขัด','สะสม','ท้ายแถว']
place_word = ['วิภาวดี',
 'แคราย',
 'สะพานพระนั่งเกล้า',
 'รัตนาธิเบศร์',
 'พงษ์เพชร',
 'เกษตร',
 'งามวงศ์วาน',
 'บางเขน']
place_word_ref = place_word


# In[4]:


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

# In[5]:


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


# In[6]:


sampling=sampling_func(neg_word,1)


# In[ ]:


query


# In[7]:


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


# In[8]:


tweets_df_q = tweets_df_q.drop(['date_time','id'],axis=1)
#tweets_df_q=tweets_df_q[tweets_df_q['tweet'].str.contains("Trump")] #Change Everytime!!!


# In[9]:


tweets_df_q.drop_duplicates(keep="first", inplace=True) #Drop duplicated items


# # Pre processing Data

# In[10]:


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


# In[11]:


nltk.download('words') #pull thai word(Bags)
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()


# In[12]:


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


# # Create Live news Dataframe

# In[13]:


#Set New dataframe by query
df = tweets_df_q


# In[14]:


list_clean_df = [clean_msg(i) for i in df.tweet] #clean msg
list_token_df = [split_word(text) for text in list_clean_df] #split words


# In[15]:


#Add token to df
df['token'] = list_token_df


# In[16]:


label_total = []
for i in range(len(df.token)):
    label = [txt for txt in df.token.iloc[i] if txt in neg_word]
    label = list(set(label))
    label_total.append(label)
    print(i)
    print(label)


# In[17]:


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
            aa[i]= 'รัตนาธิเบศร์'
        elif aa[i]=='พงศ์'or aa[i]=='เพชร':
            aa[i]='พงศ์เพชร'
        elif aa[i]=='งาม'or aa[i]=='วงศ์'or aa[i]=='วาน':
            aa[i]='งามวงศ์วาน'
        elif aa[i]=="แจ้ง" or aa[i]=='วัฒนะ':
            aa[i]='แจ้งวัฒนะ'
        else:
            pass


# In[18]:


df['condition'] = label_total #gern condition to df
df['place'] = place_total #gern place to df


# In[19]:


df


# # Modeling

# In[20]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import io
#from google.colab import files (working vai offline conda prompt)


# In[21]:


#Construct Dataframe import Data from 'colab_model.csv'
df_model = pd.read_csv('COLAB_MODEL.csv')
df_model = df_model[['nameDAY','CAL']]


# In[22]:


#Calculate Arrival Time
df.values.tolist()
nameDAY = df_model['nameDAY'].tolist()
CAL = df_model['CAL'].tolist() 

# count element
countmon = nameDAY.count('จันทร์')
counttue = nameDAY.count('อังคาร')
countwed = nameDAY.count('พุธ')
countthu = nameDAY.count('พฤหัสบดี')
countfri = nameDAY.count('ศุกร์')

li = df_model.values.tolist() 

tup = {i:0 for i, v in li}
for key, value in li:
    tup[key] = tup[key]+value
result = list(map(tuple, tup.items()))


# In[23]:


for (x,y) in result:
    if x == 'จันทร์':
        mon = y/countmon    #AVG_MONDAY
       # print('mon',int(mon))
    elif x == 'อังคาร':
        tue = y/counttue  #AVG_TUESDAY
        #print('tue' , int(tue))
    elif x == 'พุธ':
        wed = y/countwed  #AVG_WEDNESDAY
        #print('wed' ,int(wed))
    elif x == 'พฤหัสบดี':
        thu = y/countthu  #AVG_THURSDAY
        #print('thu' ,int(thu))
    elif x == 'ศุกร์':
        fri = y/countfri  #AVG_FRIDAY
        #print('fri' ,int(fri))


# In[24]:


#Workdays dataframe set up & working on apriori algo
col_name = ['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก','วิภาวดี','แคราย','สะพานพระนั่งเกล้า','รัตนาธิเบศ','พงษ์เพชร','บางเขน','เกษตร','งามวงศ์วาน']
#Monday
df_mon = pd.read_csv('COLAB_MODEL_mon.csv')
df_mon.drop('nameDAY',inplace=True,axis=1)
df_mon.columns = col_name
df_mon.drop(['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก'],inplace=True,axis=1)
frequent_itemsets_mon = apriori(df_mon, min_support=0.07, max_len=2, use_colnames=True)

#Tuesday
df_tue = pd.read_csv('COLAB_MODEL_tue.csv')
df_tue.columns = col_name
df_tue.drop(['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก'],inplace=True,axis=1)
frequent_itemsets_tue = apriori(df_tue, min_support=0.07,max_len=2, use_colnames=True)

#Wednesday
df_wed = pd.read_csv('COLAB_MODEL_wed.csv')
df_wed.columns = col_name
df_wed.drop(['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก'],inplace=True,axis=1)
frequent_itemsets_wed = apriori(df_wed, min_support=0.07,max_len=2 , use_colnames=True)

#Thursday
df_thu = pd.read_csv('COLAB_MODEL_thu.csv')
df_thu.columns = col_name
df_thu.drop(['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก'],inplace=True,axis=1)
frequent_itemsets_thu = apriori(df_thu, min_support=0.07,max_len =2, use_colnames=True)

#Friday
df_fri = pd.read_csv('COLAB_MODEL_fri.csv')
df_fri.columns = col_name
df_fri.drop(['รถติด','อุบัติเหตุ','ซ่อม','ฝนตก'],inplace=True,axis=1)
frequent_itemsets_fri = apriori(df_fri, min_support=0.07,max_len=2, use_colnames=True)


# In[25]:


#Association Rules and filter life stat >5 with >= 0.8 confidence level
rules_mon = association_rules(frequent_itemsets_mon, metric="lift", min_threshold=1)
rules_mon = rules_mon[(rules_mon['lift']>=5) & rules_mon['confidence']>=0.8] #Monday
rules_tue = association_rules(frequent_itemsets_tue, metric="lift", min_threshold=1)
rules_tue = rules_tue[(rules_tue['lift']>=3)&rules_tue['confidence']>=0.8] #Tuesday
rules_wed = association_rules(frequent_itemsets_wed, metric="lift", min_threshold=1)
rules_wed = rules_wed[(rules_wed['lift']>=6)&rules_wed['confidence']>=0.8] #Wednesday
rules_thu = association_rules(frequent_itemsets_thu, metric="lift", min_threshold=1)
rules_thu = rules_thu[(rules_thu['lift']>=2)&rules_thu['confidence']>=0.5] #Thursday
rules_fri = association_rules(frequent_itemsets_fri, metric="lift", min_threshold=1)
rules_fri = rules_fri[(rules_fri['lift']>=5)&rules_fri['confidence']>=0.8] #Friday


# In[26]:


rules_mon


# In[27]:


rules_tue


# In[28]:


rules_wed


# In[29]:


rules_fri


# In[30]:


df


# # Notify

# In[46]:


def Lineconfig(command):
	url = 'https://notify-api.line.me/api/notify'
	token = '8c4Do0dvhOAa5YRx0skmzphfO1KpF1coWDHFYgeI6Z6' ## EDIT
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

def sticker(sticker_id,package_id,message=' '):
	command = {'message':message,'stickerPackageId':package_id,'stickerId':sticker_id}
	return Lineconfig(command)

def sendnews(news):
	# send news
	command = {'message':news}
	return Lineconfig(command)


# In[32]:


#define time(arrival time) and notify function with each day
day = date.today().weekday()
if day == 0:
    time = int(mon)
    #aviod_place = [set(i)for i in rules_mon.consequents] #from a model suggestion
elif day == 1:
    time = int(tue)
    #aviod_place = [set(i)for i in rules_tue.consequents]
elif day == 2:
    time = int(wed)
    #aviod_place = [set(i)for i in rules_wed.consequents]
elif day == 3:
    time = int(thu)
    #aviod_place = [set(i)for i in rules_thu.consequents]
elif day == 4:
    time = int(fri)
    #aviod_place = [set(i)for i in rules_fri.consequents]
else:
    time="Weekend"


# In[33]:


#Construct list of condition, place, avoid_place(no duplicate)
condition_ls = [] #df.condition
place_ls = [] #df.place
for num in range(0,len(df.condition)):
    for txt in df.condition.iloc[num]:
        condition_ls.append(txt)
condition_ls = set(condition_ls)
for num in range(0,len(df.place)):
    for txt in df.place.iloc[num]:
        place_ls.append(txt)
place_ls = set(place_ls)


# In[34]:


#flatten list
def flatten(l):
    flatList = []
    for elem in l:
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList


# In[35]:


#map df.place with rules_day(dataframe)
def map_place(antecedent): #where you want to map
    map_list =[]
    con_list = []
    try:
        for text in antecedent: con_list.append(list(text))
        con_list = flatten(con_list)
        for txt in con_list:
            if txt in place_ls: #compare to place in scraped news; If it is found,keep. Unless, not keep!!
                map_list.append(txt)
                return map_list
    except BaseException as e:
        print('failed on_status,',str(e))
        time.sleep(3)


# In[36]:


def frozen(obj):
    try:
        frozenset(obj)
    except BaseException as e:
        print('failed on_status,',str(e))
        print('No suggestion for today!')


# In[44]:


#Execute program
def execute_notice(time,condition,place):
    sticker(3,6,"Good Morning")
    sendtext("การเดินทางวันนี้ใช้เวลาประมาณ {} นาที".format(time))
    sendcon("การจราจรโดยรวมมีปัญหา {}".format(condition))
    sendplace("เส้นทางที่มีปัญหา {}".format(place))

def execute_news(news):
    sendnews("สำหรับข่าวเพิ่มเติม {}".format(news))

def asso_rule(days):
    if days ==0:
        try:
            #show consequences
            test=rules_thu[rules_mon['antecedents'] == frozenset(map_place(rules_mon.antecedents))]
            test2 = test[['antecedents','consequents','confidence','lift']]
            sendtext("Possible traffic congession are {}".format(test2.consequents))
            sendtext("See possible place statistical numbers {}".format(test2))
        except BaseException as e:
            print('failed on_status,',str(e))
            sendtext("No Relevant data Today!!")
    elif days ==1:
        try:
            #show consequences
            test=rules_thu[rules_tue['antecedents'] == frozenset(map_place(rules_tue.antecedents))]
            test2 = test[['antecedents','consequents','confidence','lift']]
            sendtext("Possible traffic congession are {}".format(test2.consequents))
            sendtext("See statistical numbers {}".format(test2))
        except BaseException as e:
            print('failed on_status,',str(e))
            sendtext("No Relevant data Today!!")
    elif days ==2:
        try:
            #show consequences
            test=rules_thu[rules_wed['antecedents'] == frozenset(map_place(rules_wed.antecedents))]
            test2 = test[['antecedents','consequents','confidence','lift']]
            sendtext("Possible traffic congession are {}".format(test2.consequents))
            sendtext("See statistical numbers {}".format(test2))
        except BaseException as e:
            print('failed on_status,',str(e))
            sendtext("No Relevant data Today!!")
    elif days ==3:
        try:
            #show consequences
            test=rules_thu[rules_thu['antecedents'] == frozenset(map_place(rules_thu.antecedents))]
            test2 = test[['antecedents','consequents','confidence','lift']]
            sendtext("Possible traffic congession are {}".format(test2.consequents))
            sendtext("See statistical numbers {}".format(test2))
        except BaseException as e:
            print('failed on_status,',str(e))
            sendtext("No Relevant data Today!!")
    elif days ==4:
        try:
            #show consequences
            test=rules_thu[rules_fri['antecedents'] == frozenset(map_place(rules_fri.antecedents))]
            test2 = test[['antecedents','consequents','confidence','lift']]
            sendtext("Possible traffic congession are {}".format(test2.consequents))
            sendtext("See statistical numbers {}".format(test2))
        except BaseException as e:
            print('failed on_status,',str(e))
            sendtext("No Relevant data Today!!")
    else:
        sendtext("Weekend")


# In[38]:


execute_notice(time,condition_ls,place_ls)

for i in range(0,2): # 2 latest news
    execute_news(news=df.tweet.iloc[i])
    
asso_rule(day)


# In[47]:


asso_rule(day)

Monday = 0, Sunday =6def asso_rule(days):
    if days ==1:
        #show consequences
        test=rules_thu[rules_mon['antecedents'] == frozenset(map_place(rules_mon.antecedents))]
        #grap all consequents
        con_place = []
        for pla in test.consequents:
            con_place.append(list(pla))
            con_place= set(flatten(con_place))
        sendtext("Possible jammed places {}".format(con_place))
        test2 = test[['consequents','confidence','lift']]
        sendtext("See statistical numbers {}".format(test2))
    elif days ==2:
        #show consequences
        test=rules_thu[rules_tue['antecedents'] == frozenset(map_place(rules_tue.antecedents))]
        #grap all consequents
        con_place = []
        for pla in test.consequents:
            con_place.append(list(pla))
            con_place= set(flatten(con_place))
        sendtext("Possible jammed places {}".format(con_place))
        test2 = test[['consequents','confidence','lift']]
        sendtext("See statistical numbers {}".format(test2))
    elif days ==3:
        #show consequences
        test=rules_thu[rules_wed['antecedents'] == frozenset(map_place(rules_wed.antecedents))]
        #grap all consequents
        con_place = []
        for pla in test.consequents:
            con_place.append(list(pla))
            con_place= set(flatten(con_place))
        sendtext("Possible jammed places {}".format(con_place))
        test2 = test[['consequents','confidence','lift']]
        sendtext("See statistical numbers {}".format(test2))
    elif days ==4:
        #show consequences
        test=rules_thu[rules_thu['antecedents'] == frozenset(map_place(rules_thu.antecedents))]
        #grap all consequents
        #result_place = list()
        #for pla in test.consequents:
         #   result_place.append(pla)
          #  result_place= set(flatten(result_place))
        #sendtext("Possible jammed places {}".format(result_place))
        #test2 = test[['consequents','confidence','lift']]
        sendtext("See statistical numbers {}".format(test))
    elif days ==5:
        #show consequences
        test=rules_thu[rules_fri['antecedents'] == frozenset(map_place(rules_fri.antecedents))]
        #grap all consequents
        con_place = []
        for pla in test.consequents:
            con_place.append(list(pla))
            con_place= set(flatten(con_place))
        sendtext("Possible jammed places {}".format(con_place))
        sendtext("See statistical numbers {}".format(test2))
    else:
        sendtext("Weekend")#Not able to use
aviod_place = [set(i)for i in rules_fri.consequents]
aviod_place