import nltk
nltk.download('punkt')
from newspaper import Article
import datetime
import pandas as pd
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
from selenium import webdriver
import unicodedata
name_list=[]
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as plt
import json
import plotly

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os
import random
import pandas as pd
from flask import Flask, render_template , request

import matplotlib.pyplot as plt
import matplotlib
PEOPLE_FOLDER = os.path.join('static')

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
def stopwords_for_wordcount(df):
    list_key = []
    for x in df['clean_text']:
        list_key.append(x)
    comment_words = ''
    for token in list_key:
        if str(token) != "None":
            comment_words += token
    stopwords = set(STOPWORDS)
    stopwords.add("amp")
    stopwords.add("central")
    stopwords.add("islamic")
    stopwords.add("commercial")
    stopwords.add("standard")
    stopwords.add("chartered")
    stopwords.add("mr")
    stopwords.add("say")
    stopwords.add("please")
    stopwords.add("dh")
    stopwords.add("stop")
    stopwords.add("already")
    stopwords.add("bank")
    stopwords.add("abu")
    stopwords.add("dhabi")
    stopwords.add("cent")
    stopwords.add("uae")
    stopwords.add("dubai")
    stopwords.add("banks")
    stopwords.add("will")
    stopwords.add("nmc")
    stopwords.add("per")
    stopwords.add("al")
    stopwords.add("uaes")
    stopwords.add("said")
    stopwords.add("says")
    return stopwords,comment_words
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
def remove_stopwords(text, is_lower_case=False):
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    list_stop=["nfc","po","nmc","rakbank"]
    stopword_list=stopword_list+list_stop
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text
def remove_proper_nouns(doc):
    tagged_sentence = nltk.tag.pos_tag(doc.split())

    edited_sentence = [word for word, tag in tagged_sentence if tag == 'NN' or tag == 'JJ']

    text=' '.join(edited_sentence)
    return text
def sentiment_list(corpus):
    list=[]
    sd = {}
    sia=SentimentIntensityAnalyzer()
    for doc in corpus:
        pol_score = sia.polarity_scores(doc)
        pol_score['headline'] = doc
        if pol_score['compound']>0:
            list.append("Positive")
        elif pol_score['compound']==0:
            list.append("Neutral")
        else:
            list.append("Negative")
    return list
def normalize_corpus(corpus):
    normalize_corpus=[]
    for doc in corpus:
        doc = re.sub(r"\b[A-Z\.]{2,}s?\b", "", doc)
        doc = remove_stopwords(doc,is_lower_case=True)
        doc = remove_accented_chars(doc)

        doc = expand_contractions(doc)

        doc = doc.lower()
            # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_characters(doc,remove_digits=True)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        from pywsd.utils import lemmatize_sentence
        doc=lemmatize_sentence(doc)
        doc=' '.join(doc)
        doc = remove_proper_nouns(doc)
        normalize_corpus.append(doc)
    return normalize_corpus


def count_plz(word, doc):
    count1 = 0
    if word in doc:
        count1=doc.count(word)

    return count1

def national(link,z,x,source,country):
    headline_list=[]
    df = {}
    if source=="The National":
        url = 'https://thenational.ae/' + link
    elif source=="Khaleej Times":
        url='https://www.khaleejtimes.com/'+link
    elif source=="Egypt Today":
        url='https://www.egypttoday.com/'+link
    else:
        url=link
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    if article.title not in headline_list:
        df['country']=country
        df['source']=source
        df['bank'] = x
        df['date']=z
        df['Headline']=article.title
        headline_list.append(df['Headline'])
        df['Raw Article']=article.text
        df['Summary']=article.summary
        string=" ".join(article.keywords)
        df['Keywords']=string
        name_list.append(df)




def font_know(headline,processed_text,list_cat,str1,str2):
    DF = {}
    list_doc=processed_text.split()
    list_head=headline.split()
    list_digi=["digital services","digital platform","digital payment"]
    sum=0
    if str1 in processed_text or str2 in processed_text :
        if "Hay Festival" not in headline:
            for word in list_cat:
                count_word=count_plz(word,processed_text)
                count_word_head=count_plz(word,headline)
                prob=count_word/len(list_doc)
                prob2=count_word_head/len(list_head)
                sum+=prob+prob2


    return sum

def assign_topics(corpus,headline):
    topic_list=[]
    for doc in range(0,len(corpus)):
        list_forb=["relief","payment holiday","three month payment holiday","threemonth payment holiday","repayment holiday","deferment","mortgage payment",
                   "payment","holiday","loan","defer","fee waiver","waiver","repayment","threemonth",'support','help',"offer","relieve","three months"
                   ,"loan deferral"]
        list_contact=["digital","contactless","payment","cashless",'digital services','digital platform',"money","transfer","transaction","app","remit"
                      'mastercard',"contactless payment","contactless cashless payment","cashless payment"]
        list_credit_inc=["credit line","credit limit","credit card limit","credit card","credit limit increase","credit line increase",
                     "decrease","increase","card limit","credit line decrease","credit limit decrease"]
        list_finan=["financial report","financial statement","financial status","annual report","cash flow statements",
                    "balance sheet","business report","business result","financial result"]
        list_finan=["financial result","quarter","profit","statement","financial report","business report"]
        list_cust=["customer"]
        list_sup=["support","stimulus package","relief package"]
        list_bus=["business"]
        freq_forb=font_know(headline[doc],corpus[doc],list_forb,"payment holiday","defer loan")
        freq_cont=font_know(headline[doc],corpus[doc],list_contact,"contactless ","digital")
        freq_cred_inc=font_know(headline[doc],corpus[doc],list_credit_inc,"credit card limit","credit limit increase")
        freq_cust=font_know(headline[doc],corpus[doc],list_cust,"customer","bank")
        freq_sup=font_know(headline[doc],corpus[doc],list_sup,"support","bank")
        print(headline[doc])
        list_freq=[freq_cust,freq_forb,freq_cred_inc,freq_sup,freq_cont]
        if (max(list_freq)==freq_forb and freq_forb>=0.05)  :
            topic="Forbearance"
            print(freq_forb)
        elif max(list_freq)==freq_cont and freq_cont>=0.05  :
            topic="Contactless/Digital"
        elif freq_cred_inc>=0.05 and max(list_freq)==freq_cred_inc :
            topic="Credit Limit"
        elif freq_cust>=0.05 and max(list_freq)==freq_cust:
            topic="Customer"
        elif freq_sup>=0.05 and max(list_freq)==freq_sup:
            topic="Support"
        else:
            if any(list_finan):
                topic="Financial Result"
            elif "business" in corpus[doc] :
                topic="Business"
            else:
                topic="Others"
        topic_list.append(topic)
    return topic_list
def national_source_url(list_bank,headers):
    import csv
    for x in list_bank:
        f=0
        print("NATIONAL SOURCE URL")
        print(x)
        for i in range(1,25):
            print(i)
            url="https://www.thenational.ae/search?q={}&fq=&page={}".format(x,i)
            xb=x
            if '+' in x:
                xb=x.replace('+',' ')
                if 'Citi' in xb:
                    xb=xb.replace(" ","")
            response = requests.get(url, headers=headers)
            content = response.content
            soup = BeautifulSoup(content, "html.parser")
            list_tr = soup.find_all("div", attrs={"class": "small-article-desc $sectionColour"})
            for tr in list_tr:
                x1 = (tr.find('a'))
                title=x1.h2.text
                print(title)

                link = (x1.get('href'))
                date_p = (x1.em.text)
                print(date_p)
                date_f = datetime.datetime.strptime(date_p, '%B %d, %Y').strftime('%Y/%m/%d')
                z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
                dates=[]

                date_author = datetime.datetime(2020, 3, 15)
                if z >= date_author and z <= datetime.datetime.now():
                    national(link,z,xb,'The National',"UAE")
                else:
                    f=1
                    break
            if f==1:
                break


def arabian(driver):
    driver.implicitly_wait(100)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.implicitly_wait(200)
    result=driver.find_element_by_xpath('''//*[@id="subsection-results"]/div[300]''')
    driver.implicitly_wait(200)
    list = []
    list_bank = ['Mashreq', 'Rakbank', 'HSBC', 'First Abu Dhabi Bank', 'Abu Dhabi Commercial Bank', 'Emirates NBD',

                 'Abu Dhabi Islamic Bank', 'Standard Chartered Bank', 'Citibank']
    for x in list_bank:
        list.append(x.upper())
    print(list_bank)
    f=0
    for i in range(1, 250):
        x_path = '''//*[@id="subsection-results"]/div[{0}]'''.format(i)
        results = driver.find_element_by_xpath(x_path)
        bankname1 = results.find_element_by_tag_name('span')
        name = bankname1.find_element_by_tag_name('a')
        driver.implicitly_wait(100)
        bankname=name.text
        print(bankname)
        if (bankname) in list:
            results = (results.find_element_by_tag_name('h3'))
            res = results.find_element_by_tag_name('a')
            url=res.get_attribute('href')
            title=res.text
            response = requests.get(url, headers=headers)
            content = response.content
            soup = BeautifulSoup(content, "html.parser")

            list_tr = soup.find_all("div", attrs={"class": "date-time date-published change-font-size"})
            for tr in list_tr:
                date=(tr.span.text)

            date_f = datetime.datetime.strptime(date, '%a %d %b %Y %I:%M %p').strftime('%Y/%m/%d')

            if bankname=="HSBC":
                bankname="HSBC"
            elif bankname=="EMIRATES NBD":
                bankname="Emirates NBD"
            else:
                bankname=bankname.title()
            z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
            dates = []
            date_author = datetime.datetime(2020, 3, 15)
            print(z)
            if z >= date_author and z <= datetime.datetime.now():
                national(url, z, bankname, 'Arabian Business',"UAE")
            else:
                f=1
                break
        if f==1:
            break
def egypt(headers):
    url="https://www.egypttoday.com/Article/Search?title=nbe"
    response = requests.get(url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, "html.parser")
    f=0
    list_tr = soup.find_all("div", attrs={"class": "top-reviews-item col-xs-12 search-item article"})
    for tr in list_tr:
        x = tr.a
        link=(x.get('href'))
        title=x.text.strip()
        date_p = tr.span.text.strip()
        date_f = datetime.datetime.strptime(date_p, '%a, %b. %d, %Y').strftime('%Y/%m/%d')
        z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
        dates = []

        date_author = datetime.datetime(2020, 3, 15)

        if z >= date_author and z <= datetime.datetime.now():
            national(link, z, "National Bank of Egypt", 'Egypt Today',"Egypt")
        else:
            break


def turkey(headers):
    f=0
    for i in range(1,10):
        url="https://www.dailysabah.com/search?query=ziraat%20bank&pgno={}".format(i)
        response = requests.get(url, headers=headers)
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        list_tr = soup.find_all("div", attrs={"class": "widget_content"})

        for tr in list_tr:
            x = tr.find("a")
            title=x.text.strip()
            link=(x.get('href'))
            date_p = tr.find("div", attrs={"class": "date_text"}).text.strip()
            date_f = datetime.datetime.strptime(date_p, '%b %d, %Y').strftime('%Y/%m/%d')
            z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
            dates = []
            date_author = datetime.datetime(2020, 3, 15)

            if z >= date_author and z <= datetime.datetime.now():
                national(link, z, "Ziraat Bank", 'Daily Sabah',"Turkey")
            else:
                f=1
                break
        if f==1:
            break



def khaleej_source_url(list_bank,headers):
    import csv
    for bank in list_bank:
        f=0
        print("KHALEEJ SOURCE URL")
        print(bank)
        for i in range(1,25):
            print(i)
            url="https://www.khaleejtimes.com/search?text={}&pagenumber={}".format(bank,i)
            bank1=bank
            if '+' in bank:
                bank1=bank.replace('+',' ')
                if 'Citi' in bank1:
                    bank1=bank1.replace(" ","")
            response = requests.get(url, headers=headers)
            content = response.content
            soup = BeautifulSoup(content, "html.parser")

            list_tr = soup.find_all("div", attrs={"class":"search_listing"})

            for tr in list_tr:
                x=tr.find_all("li")
                for y in x:
                    x1 = (y.find('a'))
                    title=x1.text
                    link = (x1.get('href'))
                    date_p = y.find("div", attrs={"class": "author_date"}).text
                    date_f = datetime.datetime.strptime(date_p, '%d %B, %Y').strftime('%Y/%m/%d')
                    z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
                    dates = set()
                    date_author=datetime.datetime(2020, 3, 15)

                    if z >= date_author and z <= datetime.datetime.now():

                        national(link, z, bank1, 'Khaleej Times',"UAE")
                    else:
                        f = 1
                        break
                if f == 1:
                    break


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
list_bank=['Mashreq','Rakbank','HSBC','First+Abu+Dhabi+Bank','Abu+Dhabi+Commercial+Bank','Emirates+NBD',
'Abu+Dhabi+Islamic+Bank','Standard+Chartered+Bank','Citibank','Citi+bank']
#list_bank=['Citibank','Citi+bank']

driver = webdriver.Chrome(executable_path='C:\\Users\\Mansi Dhingra\\Downloads\\chromedriver.exe')

HOME_PAGE_URL = "https://www.arabianbusiness.com/industries/banking-finance"

driver.implicitly_wait(30)
driver.get(HOME_PAGE_URL)
driver.implicitly_wait(30)
print("**************************Arabian*************************************")
arabian(driver)
print("***********************National***************************")
national_source_url(list_bank,headers)
khaleej_source_url(list_bank,headers)
turkey(headers)

egypt(headers)

news_df=pd.DataFrame(name_list)

news_df['full_text'] = news_df["Raw Article"]
news_df['clean_text'] = normalize_corpus(news_df['full_text'])
news_df['clean_title']=normalize_corpus(news_df['Headline'])
news_df['Topic'] = assign_topics(news_df['clean_text'],news_df['Headline'])
news_df['Sentiment']=sentiment_list(news_df['clean_title'])
norm_corpus = list(news_df['clean_text'])


news_df.to_csv('news_articles.csv',index=False)

#news_df.to_csv('news_articles.csv',index=False)
