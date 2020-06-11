import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from flask import Flask, render_template, request,redirect,url_for
app = Flask(__name__)
import difflib
tl=[]
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

import pandas as pd
from collections import Counter
images=[]
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import random
random_number=random.randint(0,99999)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


news_df = pd.read_csv('menat_news_articles.csv')
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as plt
import json
import plotly

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

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def stopwords_for_wordcloud(corpus):
    list_key = []
    for x in corpus:
        list_key.append(x)
    comment_words = ''
    for token in list_key:
        if str(token) != "None":
            comment_words += token

    stopword_list = nltk.corpus.stopwords.words('english')
    #add multiple list of stop
    list=["march","week","number",'dhbn',"UAE","amp","central","islamic","standard","chartered","mr","please",'charter',
    "dh","stop","client","bank","abu","month","company","country","business","commercial","singapore","oil","asia",
    "price","dhabi",'use',"cent","uae","dubai","banks","per","al","time","year","uaes","new",'bn',"many","part","day"]
    stopword_list+=(list)
    return stopword_list,comment_words

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
    #list_stop=["nfc","po","nmc","rakbank"]
    #stopword_list=stopword_list+list_stop
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




def find_prob(headline,processed_text,list_cat,str1,str2):
    list_doc=processed_text.split()
    list_head=headline.split()
    sum=0
    if str1 in processed_text or str2 in processed_text :
        for word in list_cat:
            count_word=processed_text.count(word)
            count_word_head=headline.count(word)
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
                     "increase","card limit","credit line decrease","credit limit decrease"]
        list_cust=["customer"]
        list_sup=["support","stimulus package","relief package","package","economic package"]
        freq_forb=find_prob(headline[doc],corpus[doc],list_forb,"payment holiday","defer loan")
        freq_cont=find_prob(headline[doc],corpus[doc],list_contact,"contactless ","digital")
        freq_cred_inc=find_prob(headline[doc],corpus[doc],list_credit_inc,"credit card limit","credit limit increase")
        freq_cust=find_prob(headline[doc],corpus[doc],list_cust,"customer","bank")
        freq_sup=find_prob(headline[doc],corpus[doc],list_sup,"support","bank")
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

            list_finan = ["financial result", "financial report", "business report", "financial statement"]
            if any(word in corpus[doc] for word in list_finan):
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
                df = pd.read_csv('menat_news_articles.csv')
                for ind in df.index:
                    if df['source'][ind] == "The National" and df['bank'][ind] == xb:
                        dates.append(df['date'][ind])
                title_list=[]
                if len(dates) > 0:
                    date_p=max((dates))
                    title_list=(df[df['date'] == date_p]['Headline']).to_list()
                    date_f1 = datetime.datetime.strptime(date_p, '%Y-%m-%d').strftime('%Y/%m/%d')
                    z1 = datetime.datetime.strptime(date_f1, '%Y/%m/%d')
                    date_author = z1
                else:
                    date_author = datetime.datetime(2020, 3, 15)
                print(title_list)
                if z >= date_author and z <= datetime.datetime.now():
                    if title not in title_list:

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
    result=driver.find_element_by_xpath('''//*[@id="subsection-results"]/div[100]''')

    list = []
    list_bank = ['Mashreq', 'Rakbank', 'HSBC', 'First Abu Dhabi Bank', 'Abu Dhabi Commercial Bank', 'Emirates NBD',

                 'Abu Dhabi Islamic Bank', 'Standard Chartered Bank', 'Citibank']
    for x in list_bank:
        list.append(x.upper())
    print(list_bank)
    for i in range(1, 100):
        x_path = '''//*[@id="subsection-results"]/div[{0}]'''.format(i)
        results = driver.find_element_by_xpath(x_path)
        bankname1 = results.find_element_by_tag_name('span')
        name = bankname1.find_element_by_tag_name('a')
        driver.implicitly_wait(100)
        bankname=name.text
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
            df = pd.read_csv('menat_news_articles.csv')
            for ind in df.index:
                if df['source'][ind] == "Arabian Business" and df['bank'][ind] == bankname:
                    dates.append(df['date'][ind])
            title_list=[]
            if len(dates) > 0:
                date_p = (max(dates))
                title_list=(df[df['date'] == date_p]['Headline']).to_list()
                date_f1 = datetime.datetime.strptime(date_p, '%Y-%m-%d').strftime('%Y/%m/%d')
                z1 = datetime.datetime.strptime(date_f1, '%Y/%m/%d')
                date_author = z1
            else:
                date_author = datetime.datetime(2020, 3, 15)

            if z >= date_author and z <= datetime.datetime.now():
                if title not in title_list:
                    print(bankname)
                    national(url, z, bankname, 'Arabian Business',"UAE")
            else:
                break

def egypt(headers):
    url="https://www.egypttoday.com/Article/Search?title=nbe"
    response = requests.get(url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, "html.parser")
    list_tr = soup.find_all("div", attrs={"class": "top-reviews-item col-xs-12 search-item article"})
    for tr in list_tr:
        x = tr.a
        link=(x.get('href'))
        title=x.text.strip()
        date_p = tr.span.text.strip()
        date_f = datetime.datetime.strptime(date_p, '%a, %b. %d, %Y').strftime('%Y/%m/%d')
        z = datetime.datetime.strptime(date_f, '%Y/%m/%d')
        dates = []
        df = pd.read_csv('menat_news_articles.csv')
        for ind in df.index:
            if df['source'][ind] == "Egypt Today" and df['bank'][ind] == "National Bank of Egypt":
                dates.append(df['date'][ind])
        title_list=[]
        if len(dates) > 0:
            date_p = (max(dates))
            title_list=(df[df['date'] == date_p]['Headline']).to_list()
            date_f1 = datetime.datetime.strptime(date_p, '%Y-%m-%d').strftime('%Y/%m/%d')
            z1 = datetime.datetime.strptime(date_f1, '%Y/%m/%d')
            date_author = z1
        else:
            date_author = datetime.datetime(2020, 3, 15)

        if z >= date_author and z <= datetime.datetime.now():
            if title not in title_list:

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
            df = pd.read_csv('menat_news_articles.csv')
            for ind in df.index:
                if df['source'][ind] == "Daily Sabah" and df['bank'][ind] == "Ziraat Bank":
                    dates.append(df['date'][ind])
            title_list=[]
            if len(dates) > 0:
                date_p = (max(dates))
                title_list=(df[df['date'] == date_p]['Headline']).to_list()
                date_f1 = datetime.datetime.strptime(date_p, '%Y-%m-%d').strftime('%Y/%m/%d')
                z1 = datetime.datetime.strptime(date_f1, '%Y/%m/%d')
                date_author = z1
            else:
                date_author = datetime.datetime(2020, 3, 15)

            if z >= date_author and z <= datetime.datetime.now():
                if title not in title_list:

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
                    df = pd.read_csv('menat_news_articles.csv')
                    for ind in df.index:
                        if df['source'][ind] == "Khaleej Times" and df['bank'][ind] == bank1:
                            dates.add(df['date'][ind])
                    title_list=[]
                    if len(dates) > 0:
                        date_p = (max(dates))
                        title_list = (df[df['date'] == date_p]['Headline']).to_list()
                        date_f1 = datetime.datetime.strptime(date_p, '%Y-%m-%d').strftime('%Y/%m/%d')
                        z1 = datetime.datetime.strptime(date_f1, '%Y/%m/%d')
                        date_author = z1
                    else:
                        date_author=datetime.datetime(2020, 3, 15)
                    print(title_list)
                    if z >= date_author and z <= datetime.datetime.now():
                        if title not in title_list:
                            print(bank1)
                            national(link, z, bank1, 'Khaleej Times',"UAE")
                    else:
                        f=1
                        break
            if f==1:
                break

#Graphs for bigrams,positive,negative and sentiment pie chart
def create_graphs(x,y,string):
    if string=="bigram" or string=="unigram" or string=="trigram":
        trace1 = go.Bar(x=x, y=y, marker_color='grey')
        layout = go.Layout(title="Top 20 " + string + " words", xaxis=dict(title="Words", ),
                           yaxis=dict(title="Count", ), autosize=False, width=470, height=380,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')

    elif string=="positive":

        trace1 = go.Bar(x=x, y=y, marker_color='grey')
        layout = go.Layout(title="Top most "+string+" words", xaxis=dict(title="Words", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=380,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
    else:

        trace1 = go.Bar(x=x, y=y, marker_color='red')
        layout = go.Layout(title="Top most "+string+" words", xaxis=dict(title="Words", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=380,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig_json


def fetch_sentiment_using_vader(corpus):

    pos_word_list = set()
    word_total = []
    neg_word_list = set()

    for text in corpus:
        list_words=text.split()
        sid = SentimentIntensityAnalyzer()
        for word in list_words:
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                pos_word_list.add(word)
            elif (sid.polarity_scores(word)['compound']) <= -0.5:
                neg_word_list.add(word)
            word_total.append(word)
    return pos_word_list,neg_word_list,word_total


#Plot positive and negative bar charts
def plot_words(pos_word_list, word_total, string):
    import matplotlib.pyplot as plt
    list_count = []
    for word in pos_word_list:
        dict = {}
        dict['word'] = word
        dict['word_count'] = word_total.count(word)
        list_count.append(dict)
    newlist = sorted(list_count, key=lambda k: k['word_count'], reverse=True)[0:10]
    toplist = []
    clist = []
    for top in newlist:
        toplist.append(top['word'])
        clist.append(top['word_count'])
        print(clist)
    fig_json=create_graphs(toplist,clist,string)
    return fig_json

#Articlees across category
def count_category(df):
    s=df['Topic'].value_counts(sort=True)
    new = pd.DataFrame({'Category': s.index, 'Count': s.values})
    x=new['Category'].to_list()
    y=new['Count'].to_list()
    trace1 = go.Bar(x=x, y=y, marker_color='red')
    layout = go.Layout(title="No. of articles per category", xaxis=dict(title="Categories", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=400,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

#Articles across publication
def count_pub(df):
    s = df['source'].value_counts(sort=True)
    new = pd.DataFrame({'Publication': s.index, 'Count': s.values})
    x = new['Publication'].to_list()
    y = new['Count'].to_list()
    trace1 = go.Bar(x=x, y=y, marker_color='red')
    layout = go.Layout(title="No. of articles per publication", xaxis=dict(title="Publications", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=400,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

#Sentiment count pie chart
def count_sent_pie(x1,y1):
    x=[]
    y=[]
    for l in x1:
        x.append(l)
    for l1 in y1:
        y.append(l1)
    colors = ["red", "grey", "black"]
    trace1 = go.Pie(labels=x,
                    values=y,
                    hoverinfo='label+value+percent'
                    )
    layout = go.Layout(title="Sentiment Counts", autosize=False, width=380, height=380)
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(marker=dict(colors=colors))
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

#Articles across banks
def count_bank(df):
    s = df['bank'].value_counts(sort=True)
    new = pd.DataFrame({'Banks': s.index, 'Count': s.values})
    x = new['Banks'].to_list()
    y = new['Count'].to_list()
    trace1 = go.Bar(x=x, y=y, marker_color='black')
    layout = go.Layout(title="No. of articles across bank", xaxis=dict(title="Banks", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=400,
                       paper_bgcolor='rgba(0,0,0,0)',margin=dict(b=180,pad=8),
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json
def count_country(df):
    s = df['country'].value_counts(sort=True)
    new = pd.DataFrame({'Country': s.index, 'Count': s.values})
    x = new['Country'].to_list()
    y = new['Count'].to_list()
    trace1 = go.Bar(x=x, y=y, marker_color='black')
    layout = go.Layout(title="No. of articles across country", xaxis=dict(title="Countries", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=400,
                       paper_bgcolor='rgba(0,0,0,0)', margin=dict(b=180, pad=8),
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json
def create_text(title,results):
    if title:
        with open('static/text/'+title+'.txt', 'w',encoding="utf-8") as f:
            f.write("%s\n" % str(results))

def create_summary(title, results):
    if title:
        with open('static/summaries/' + title + '.txt', 'w', encoding="utf-8") as f:
            f.write("%s\n" % str(results))


def create_wordcloud(stopwords):
    import matplotlib.pyplot as plt
    def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

    wordcloud = WordCloud(width=1200, height=1200,
                          background_color='black',
                          stopwords=stopwords[0], max_words=800,
                          min_font_size=10).generate(stopwords[1])

    # change the color setting
    wordcloud.recolor(color_func=grey_color_func)
    f7 = plt.figure(6,  facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    random_number = random.randint(0, 99999)
    name = 'wordcloud' + str(random_number) + '.png'
    plt.savefig("static/images/"+name)
    images.append(name)


def bigram_or_trigram(corpus,stopwords,string):
    def get_top_n_bigram(corpus,string, n=None):
        vec = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords[0]).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    common_words = get_top_n_bigram(corpus, string,20)
    list_words=[]
    list_freq=[]
    for word, freq in common_words:
        list_words.append(word)
        list_freq.append(freq)
    fig_json=create_graphs(list_words,list_freq,string)
    return fig_json

#grouped sentiment chart for the home tab
def create_sentiment_grouped(sent_topic,news_df):
    df = {}
    i = 0
    colors = ["red", "grey", "black"]
    for sent in sent_topic:
        topic_count=news_df[news_df['Sentiment']==sent][['Topic']].groupby('Topic').size()
        new = pd.DataFrame({'Topic': topic_count.index, 'Count': topic_count.values})
        x = new['Topic'].to_list()
        y = new['Count'].to_list()
        df[i] = go.Bar(name=sent, x=x, y=y, marker_color=colors[i])
        i = i + 1
    layout = go.Layout(title="Sentiments per Category", xaxis=dict(title="Category", ),
                       yaxis=dict(title="Count", ), autosize=False, width=500, height=380,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [df[0], df[1], df[2]]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig.update_layout(barmode='group')

    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

#resultant df for the filtered conditions
def find_resultant_df(column,list,news_df):
    final = []
    column1 = []
    f = 0
    for i in range(0, len(list)):
        if list[i] != "All":
            f = 1
            print(list[i])
            final.append(list[i])
            column1.append(column[i])
    if f == 0:
        df1 = news_df
    for x in range(0, len(column1)):
        if x == 0:
            df1 = news_df[news_df[column1[x]] == final[x]]
        else:
            df1 = df1[df1[column1[x]] == final[x]]
    return df1

#create resultant df for search article directory
def resultant_df(query,query3):
    df1 = pd.DataFrame(query, columns=['Date', 'Country', 'Publication', 'Bank', 'Category', 'Sentiment', 'Title',
                                       'Clean title'])
    df1 = (df1.drop(['Clean title'], axis=1))
    df2 = pd.DataFrame(query3, columns=['Clean title', 'Raw Article', 'Summary'])
    df2 = (df2.drop(['Clean title'], axis=1))
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
    print(df)
    return df

#flask application started and displays a landing page
@app.route('/')
def first():
    return render_template("first_news.html")

#scraping refresh and home page
@app.route('/home',methods=['get','post'])
def home():
    #empty the images folder every time it is reloaded
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    #if you want to refresh the whole dashboard with newer articles, click the refresh button
    if request.form.get('refresh'):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
        list_bank = ['Mashreq', 'Rakbank', 'HSBC', 'First+Abu+Dhabi+Bank', 'Abu+Dhabi+Commercial+Bank', 'Emirates+NBD',
                      'Abu+Dhabi+Islamic+Bank', 'Standard+Chartered+Bank', 'Citibank', 'Citi+bank']
        driver = webdriver.Chrome(executable_path='C:\\Users\\Mansi Dhingra\\Downloads\\chromedriver.exe')
        HOME_PAGE_URL = "https://www.arabianbusiness.com/industries/banking-finance"
        driver.implicitly_wait(30)
        driver.get(HOME_PAGE_URL)
        driver.implicitly_wait(30)
        arabian(driver)
        print("ARABIAN")
        national_source_url(list_bank, headers)
        print("THE NATIONAL")
        khaleej_source_url(list_bank, headers)
        print("KHALEEJ TIMES")
        turkey(headers)
        print("TURKEY")
        egypt(headers)
        print("EGYPT")
        if len(name_list)>0:
            news_df = pd.DataFrame(name_list)
            news_df['full_text'] = news_df["Raw Article"]
            news_df['clean_text'] = normalize_corpus(news_df['full_text'])
            news_df['clean_title'] = normalize_corpus(news_df['Headline'])
            news_df['Topic'] = assign_topics(news_df['clean_text'], news_df['Headline'])
            news_df['Sentiment'] = sentiment_list(news_df['clean_title'])
            news_df.to_csv('menat_news_articles.csv', mode='a', header=False, index=False)
        #once your csv file is updated redriect to the same page with fresh data
        return redirect(url_for("home"))

    news_df = pd.read_csv('menat_news_articles.csv')
    # length pf the dataframes in terms of rows
    rows = len(news_df.axes[0])
    # if you want to search something from the home page itself, just write in the search box
    search = request.form.get("search")
    #List of Category from the dataframe
    topic_list=news_df.Topic.unique()
    country_list = news_df.country.unique()
    source_list = news_df.source.unique()
    bank_list=news_df.bank.unique()
    max_date=news_df['date'].max()
    sent_topic_list=news_df.Sentiment.unique()
    sent_list=news_df.Sentiment
    #sentiment count per category
    fig_json=create_sentiment_grouped(sent_topic_list,news_df)
    #sentiment count overall pie chart
    fig_sent=count_sent_pie(Counter(sent_list).keys(), Counter(sent_list).values())
    print(fig_sent)
    #list of positive negative and total words of the cleaned article
    list_words = fetch_sentiment_using_vader(news_df['clean_text'])
    #wordcloud
    stopwords = stopwords_for_wordcloud(news_df['clean_text'])
    create_wordcloud(stopwords)
    #graph for positive words
    fig_pos=plot_words(list_words[0], list_words[2], "positive")
    #graph for negative words
    fig_neg=plot_words(list_words[1], list_words[2], "negative")
    #graph forno. of categories
    fig_cat=count_category(news_df)
    #graph for publications
    fig_pub=count_pub(news_df)
    #graph for banks
    fig_bank=count_bank(news_df)
    fig_cont=count_country(news_df)
    images_list = os.listdir(os.path.join(app.static_folder, "images"))
    return render_template('news_home.html',rows=rows,fig_pub=fig_pub,topic_list=topic_list,img=images_list,plt_pos=fig_pos,plt_neg=fig_neg,
                           bank_list=bank_list,fig_json=fig_json,source_list=source_list,max_date=max_date,fig_cat=fig_cat,fig_cont=fig_cont,
                           fig_sent=fig_sent,search=search,fig_bank=fig_bank,sent_topic=sent_topic_list,country_list=country_list)

@app.route('/category',methods=["get","post"])
def filter_func():

    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/text'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/summaries'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    search = request.form.get("search")
    news_df = pd.read_csv('menat_news_articles.csv')

    #creating list for filters for this particular html page
    topic_list = news_df.Topic.unique()
    country_list = news_df.country.unique()
    source_list = news_df.source.unique()
    bank_list = news_df.bank.unique()
    max_date = news_df['date'].max()
    sent_topic_list=news_df.Sentiment.unique()

    #fetch results from the filters
    country_result=request.form["country_list"]
    result=request.form["topic_list"]
    source_result=request.form["source_list"]
    bank_result=request.form["bank_list"]
    sent_result=request.form["sent_list"]
    start_date=request.form["start_date"]
    end_date=request.form["end_date"]

    #create resultant df based on the values selected in the filter
    column=['country','Topic','source','bank','Sentiment']
    list=[country_result,result,source_result,bank_result,sent_result]
    df1=find_resultant_df(column,list,news_df)
    #check if the resultant df exists within the dates selected
    df1 = df1[(df1['date'] >= start_date)]
    df1 = df1[(df1['date'] <= end_date)]
    df1 = df1.sort_values(by=['date'])

    clean_text_list = df1.clean_text
    sent_list=df1.Sentiment

    for row in df1.index:
        if pd.isna(df1['clean_title'][row]):
            create_text("file", df1["Raw Article"][row])
            create_summary("file", df1['Summary'][row])
        create_text(df1['clean_title'][row],df1["Raw Article"][row])
        create_summary(df1['clean_title'][row],df1['Summary'][row])

    #print the statement when the filters lead to no results
    if len(clean_text_list)==0:
        string="No results found."
        return render_template("index_news.html",string=string,topic_list=topic_list,result=result,source_result=source_result,sent_topic=sent_topic_list,
                           source_list=source_list,bank_list=bank_list,bank_result=bank_result,start_date=start_date,end_date=end_date
                               ,country_list=country_list)
    # create table to export,fetch all the data that was asked with the filters store it on a excel file and export whenever user wants
    list_columns=['date','country', 'source', 'bank', 'Topic' ,'Sentiment' ,'Headline','clean_text','clean_title',"Raw Article",'Summary']
    df2 = df1[list_columns]
    df2 = (df2.drop(['clean_text', 'clean_title'], axis=1))
    list_columns1=["Date","Country","Publication","Bank","Category","Sentiment","Title","Article","Summary"]
    df2.columns=list_columns1
    #create table to export,fetch all the data that was asked with the filters store it on a excel file and export whenever user wants
    df2.to_excel('static/table/article_directory.xlsx')
    table_list = os.listdir(os.path.join(app.static_folder, "table"))

    fig_pie = "None"
    # create a pie chart only when sentiments are filtered as all and in sent_list, we have a list of all the sentiments for every row
    if sent_result == "All":
        fig_pie = count_sent_pie(Counter(sent_list).keys(), Counter(sent_list).values())
    #create wordcloud and remove unwanted stopwords
    stopwords = stopwords_for_wordcloud(clean_text_list)
    create_wordcloud(stopwords)
    #creates bigram
    fig_tri = bigram_or_trigram(clean_text_list, stopwords, "bigram")
    #wordcloud is stored in this folder
    images_list = os.listdir(os.path.join(app.static_folder, "images"))
    #download articles and summaries content are in text and summaries folder
    text_list1 = os.listdir(os.path.join(app.static_folder, "text"))
    summary_list1=os.listdir(os.path.join(app.static_folder, "summaries"))

    return render_template("index_news.html",topic_list=topic_list,result=result,source_result=source_result,summary=summary_list1,country_result=country_result,
                           source_list=source_list,sent_topic=sent_topic_list,bank_list=bank_list,bank_result=bank_result,start_date=start_date,end_date=end_date,
                           max_date=max_date,query=df1,string="None",search=search,fig_pie=fig_pie,country_list=country_list,sent_result=sent_result,
                           fig_tri=fig_tri,img1=images_list,text1=text_list1,table_excel=table_list,list_columns=list_columns)

@app.route('/search',methods=["get","post"])
def search_func():
    result = request.args.get('search')

    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/text'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/summaries'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    search = request.form.get("search")
    news_df = pd.read_csv('menat_news_articles.csv')
    news_df.to_sql('users', con=engine)

    #because the filters also exist in the search page
    topic_list = news_df.Topic.unique()
    country_list = news_df.country.unique()
    source_list = news_df.source.unique()
    bank_list = news_df.bank.unique()
    max_date = news_df['date'].max()
    sent_topic_list=news_df.Sentiment.unique()

    query = engine.execute('''Select date,country,source,bank,Topic,Sentiment,Headline,clean_title from users where "Raw Article"
     like ('%' || ? || '%') order by date ''',(str(result),) ).fetchall()
    query2 = engine.execute('''Select "clean_text",Sentiment from users where  "Raw Article"
     like ('%' || ? || '%')''',(str(result),) ).fetchall()
    query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where  "Raw Article"
     like ('%' || ? || '%')''',(str(result),) ).fetchall()
    clean_text_list = []
    sent_list=[]
    for x in query2:
        clean_text_list.append(x[0])
        if len(x)>1:
            sent_list.append(x[1])
    if len(clean_text_list) == 0:
        string = "No results found."
        return render_template("search_news.html", string=string, topic_list=topic_list, result=result,sent_topic=sent_topic_list,
                            source_list=source_list, bank_list=bank_list, end_date=max_date,max_date=max_date,country_list=country_list)
    #wordcloud
    stopwords = stopwords_for_wordcloud(clean_text_list)
    create_wordcloud(stopwords)
    images_list = os.listdir(os.path.join(app.static_folder, "images"))

    #bigrams
    fig_tri = bigram_or_trigram(clean_text_list, stopwords, "bigram")
    for row in query3:
        create_text(row[0], row[1])
        create_summary(row[0], row[2])
    #query had some columns and query3 had some columns so needed to concatenate to make a full df
    df=resultant_df(query,query3)
    #df converted to excel for exporting
    df.to_excel('static/table/article_directory.xlsx')
    table_list = os.listdir(os.path.join(app.static_folder, "table"))
    #download articles and summary for the searched data
    text_list1 = os.listdir(os.path.join(app.static_folder, "text"))
    summary_list1 = os.listdir(os.path.join(app.static_folder, "summaries"))
    #sentiment pie chart for results
    fig_pie=count_sent_pie(Counter(sent_list).keys(),Counter(sent_list).values())
    return render_template("search_news.html", topic_list=topic_list, result=result,
                           summary=summary_list1,fig_pie=fig_pie,sent_topic=sent_topic_list,
                           source_list=source_list, bank_list=bank_list,table_excel=table_list,country_list=country_list,
                           max_date=max_date, query=query, string="None", search=search,
                           fig_tri=fig_tri, img1=images_list, text1=text_list1)


if __name__ == "__main__":
    app.run(debug=True)