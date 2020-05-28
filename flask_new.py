import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from flask import Flask, render_template, request
app = Flask(__name__)
import difflib
tl=[]
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)
import requests
from bs4 import BeautifulSoup
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


news_df = pd.read_csv('news_information1.csv')
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as plt
import json
import plotly
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
    elif string=="sentiment":
        colors=["red","grey","black"]
        trace1 = go.Pie(labels=x,
                      values=y,
                      hoverinfo='label+value+percent'
                      )
        layout = go.Layout(title="Sentiment Counts", autosize=False, width=430, height=380)
    else:

        trace1 = go.Bar(x=x, y=y, marker_color='red')
        layout = go.Layout(title="Top most "+string+" words", xaxis=dict(title="Words", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=380,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    if string=="sentiment":
        fig.update_traces(marker=dict(colors=colors))
    fig.update_xaxes(tickangle=90)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig_json


def fetch_sentiment_using_vader(corpus):

    pos_word_list = []
    word_total = []
    neg_word_list = []

    for text in corpus:
        list_words=text.split()
        sid = SentimentIntensityAnalyzer()
        for word in list_words:
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                if word not in pos_word_list:
                    pos_word_list.append(word)
            elif (sid.polarity_scores(word)['compound']) <= -0.5:
                if word not in neg_word_list :
                    neg_word_list.append(word)
            word_total.append(word)
    return pos_word_list,neg_word_list,word_total
def stopwords_for_wordcount(corpus):
    list_key = []
    for x in corpus:
        list_key.append(x)
    comment_words = ''
    for token in list_key:
        if str(token) != "None":
            comment_words += token
    stopwords = set(STOPWORDS)
    stopwords.add("NMC")
    stopwords.add("march")
    stopwords.add("week")
    stopwords.add("number")
    stopwords.add('dhbn')
    stopwords.add("UAE")
    stopwords.add("nmc")
    stopwords.add("amp")
    stopwords.add("central")
    stopwords.add("islamic")
    stopwords.add("standard")
    stopwords.add("chartered")
    stopwords.add("mr")
    stopwords.add("say")
    stopwords.add("please")
    stopwords.add('charter')
    stopwords.add("dh")
    stopwords.add("stop")
    stopwords.add("client")
    stopwords.add("already")
    stopwords.add("nfc")
    stopwords.add("po")
    stopwords.add("bank")
    stopwords.add("abu")
    stopwords.add("month")
    stopwords.add("company")
    stopwords.add("country")
    stopwords.add("business")
    stopwords.add("commercial")
    stopwords.add("singapore")
    stopwords.add("oil")
    stopwords.add("asia")
    stopwords.add("price")
    stopwords.add("dhabi")
    stopwords.add('use')
    stopwords.add("cent")
    stopwords.add("uae")
    stopwords.add("dubai")
    stopwords.add("banks")
    stopwords.add("will")
    stopwords.add("nfc")
    stopwords.add("po")
    stopwords.add("nmc")
    stopwords.add("per")
    stopwords.add("al")
    stopwords.add("time")
    stopwords.add("year")
    stopwords.add("uaes")
    stopwords.add("new")
    stopwords.add("said")
    stopwords.add("says")
    stopwords.add('bn')
    stopwords.add("many")
    stopwords.add("part")
    stopwords.add("day")
    return stopwords,comment_words
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(10, 8))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=30)
    plt.tight_layout()
    plt.xlabel('words')
    plt.ylabel('counts')
    random_number=random.randint(0,99999)
    name="top_10_words"+str(random_number)+".png"
    plt.savefig("static/images/"+name)
    images.append(name)
def count_plz(word, list_words):
    count1 = 0
    for x in list_words:
        if x == word:
            count1 += 1

    return count1
def plot_words(pos_word_list, word_total, string):
    import matplotlib.pyplot as plt
    list_count = []
    for word in pos_word_list:
        dict = {}
        dict['word'] = word
        dict['word_count'] = count_plz(word, word_total)

        list_count.append(dict)
    newlist = sorted(list_count, key=lambda k: k['word_count'], reverse=True)[0:10]
    toplist = []
    clist = []
    for top in newlist:
        if str(top['word'])!="nfc" or str(top['word'])!="po":
            toplist.append(top['word'])
            clist.append(top['word_count'])
    fig_json=create_graphs(toplist,clist,string)
    return fig_json
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
@app.route('/')
def home():
    return render_template("first_news.html")

@app.route('/home',methods=["get","post"])
def showjson():
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/news/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

    news_df = pd.read_csv('news_information1.csv')
    news_df.to_sql('users', con=engine)
    topic_l = engine.execute('''Select distinct Topic from users''').fetchall()
    topic_list=[]
    rows=len(news_df.axes[0])
    for tr in topic_l:
        topic_list.append(tr[0])

    search = request.form.get("search")
    source_l=engine.execute('''Select distinct source from users''').fetchall()
    source_list = []
    for tr in source_l:
        source_list.append(tr[0])
    bank_l = engine.execute('''Select distinct bank from users''').fetchall()
    bank_list = []
    for tr in bank_l:
        bank_list.append(tr[0])
    end_date = engine.execute('''Select max(date) from users''').fetchall()

    max_date=end_date[0][0]
    sent_count = engine.execute('''Select Sentiment,Count(*) from users group by Sentiment''').fetchall()
    sent_topic = []
    sent_count1 = []
    for tx in sent_count:
        sent_topic.append(tx[0])
        sent_count1.append(tx[1])
    df={}
    i=0

    colors=["red","grey","black"]
    for sent in sent_topic:
        topic_count=engine.execute('''Select Topic,Count(*) from users where Sentiment=? group by Topic''',(sent,)).fetchall()
        print(topic_count)
        list_s=[]
        list_t=[]
        for row in topic_count:
            list_s.append(row[1])
            list_t.append(row[0])
        df[i]=go.Bar(name=sent, x=list_t, y=list_s,marker_color=colors[i])
        i=i+1

    layout = go.Layout(title="Sentiments per Category", xaxis=dict(title="Category", ),
                       yaxis=dict(title="Count", ), autosize=False, width=430, height=380,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    data = [df[0],df[1],df[2]]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=90)
    fig.update_layout(barmode='group')

    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    fig_sent=create_graphs(sent_topic,sent_count1,"sentiment")
    print(fig_sent)
    from sklearn.feature_extraction.text import CountVectorizer
    cs = CountVectorizer(max_features=15000)
    X = cs.fit_transform(news_df['clean_text']).toarray()
    print(X)
    print("*************************************8")
    list_words = fetch_sentiment_using_vader(news_df['clean_text'])

    stopwords = stopwords_for_wordcount(news_df['clean_text'])
    fig_pos=plot_words(list_words[0], list_words[2], "positive")
    print(fig_pos)
    fig_neg=plot_words(list_words[1], list_words[2], "negative")
    print(fig_neg)
    fig_cat=count_category(news_df)
    print(fig_cat)
    fig_pub=count_pub(news_df)
    print(fig_pub)
    fig_bank=count_bank(news_df)
    print(fig_bank)
    create_wordcloud( stopwords)
    fig_tri=bigram_or_trigram(news_df['clean_text'], stopwords,"bigram")
    print(fig_tri)
    images_list = os.listdir(os.path.join(app.static_folder, "images"))
    return render_template('news_home.html',rows=rows,fig_pub=fig_pub,topic_list=topic_list,img=images_list,plt_pos=fig_pos,plt_tri=fig_tri,plt_neg=fig_neg,
                           bank_list=bank_list,fig_json=fig_json,source_list=source_list,max_date=max_date,fig_cat=fig_cat,
                           fig_sent=fig_sent,search=search,fig_bank=fig_bank,sent_topic=sent_topic)
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
    news_df = pd.read_csv('news_information1.csv')
    news_df.to_sql('users', con=engine)
    topic_l = engine.execute('''Select distinct Topic from users''').fetchall()
    topic_list = []
    for tr in topic_l:
        topic_list.append(tr[0])

    source_l=engine.execute('''Select distinct source from users''').fetchall()
    source_list = []
    for tr in source_l:
        source_list.append(tr[0])
    bank_l = engine.execute('''Select distinct bank from users''').fetchall()
    bank_list = []
    for tr in bank_l:
        bank_list.append(tr[0])
    end_date=engine.execute('''Select max(date) from users''').fetchall()

    for x in end_date:
        max_date=x[0]
    result=request.form["topic_list"]
    source_result=request.form["source_list"]
    bank_result=request.form["bank_list"]
    sent_result=request.form["sent_list"]
    start_date=request.form["start_date"]
    end_date=request.form["end_date"]

    if result!="All":
        if source_result!="All" and bank_result!="All":

            string="Following results are for "+bank_result+" from "+source_result+" site."
            query=engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where Topic=? and source=? and bank=? 
             and date between ? and ?''',(str(result),str(source_result),str(bank_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where Topic=? and source=? and bank=? and date between ? and ?''',(str(result),
                            str(source_result),str(bank_result),start_date,end_date)).fetchall()
            query3=engine.execute('''Select clean_title,"Raw Article", Summary from users where  Topic=? and source=? and bank=? 
            and date between ? and ?''',(str(result),
                            str(source_result),str(bank_result),start_date,end_date)).fetchall()

        elif source_result!="All" and bank_result=="All":
            string="Following results are from "+source_result+" site."
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where Topic=? and source=? 
             and date between ? and ?''',
                                   (str(result),
                                    str(source_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where Topic=? and source=? and date between ? and ? ''',(str(result),
                            str(source_result),start_date,end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where Topic=? and source=? 
             and date between ? and ?''',(str(result),
                            str(source_result),start_date,end_date)).fetchall()
        elif source_result=="All" and bank_result!="All":
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where Topic=? and bank=? 
             and date between ? and ?''',(str(result),str(bank_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where Topic=? and bank=? and date between ? and ?''',(str(result)
                           ,str(bank_result),start_date,end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where Topic=? and bank=? and date between ? and ?
            ''',(str(result) ,str(bank_result),start_date,end_date)).fetchall()
        else:
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where Topic=? and date between ? and ?  ''',
                                   (str(result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where Topic=?  and date between ? and ?  ''',(str(result),start_date,end_date
           )).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where Topic=?   and date between ? and ?''',
                                    (str(result),start_date,end_date)).fetchall()

    else:
        if source_result!="All" and bank_result!="All":
            string="Following results are for "+bank_result+" from "+source_result+" site."
            query=engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where source=? and bank=?
               and date between ? and ?''',(
                            str(source_result),str(bank_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where source=? and bank=?  and date between ? and ? ''',(
                            str(source_result),str(bank_result),start_date,end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where source=? and bank=?  and date between ? and ? ''',(
                            str(source_result),str(bank_result),start_date,end_date)).fetchall()

        elif source_result!="All" and bank_result=="All":
            string="Following results are from "+source_result+" site."
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where source=? and date between ? and ? ''',
                                   (str(source_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where  source=?  and date between ? and ?''',(
                            str(source_result),start_date,end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where  source=? and date between ? and ?''',(
                            str(source_result),start_date,end_date)).fetchall()
        elif source_result=="All" and bank_result!="All":
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users where  bank=?  and date between ? and ? ''',
                                   (str(bank_result),start_date,end_date)).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where  bank=?  and date between ? and ?''',(
                           str(bank_result),start_date,end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where  bank=?  and date between ? and ?''',(
                           str(bank_result),start_date,end_date)).fetchall()
        else:
            query = engine.execute('''Select Sentiment,date,source,bank,Topic,Headline,clean_title from users  where date between ? and ? ''',
                                   ( start_date, end_date) ).fetchall()
            query2=engine.execute('''Select "clean_text",Sentiment from users where date between ? and ?   ''',( start_date, end_date)).fetchall()
            query3 = engine.execute('''Select clean_title,"Raw Article", Summary from users where date between ? and ? '''
                                    ,( start_date, end_date)).fetchall()

    clean_text_list = []
    sent_list=[]

    sent_count = engine.execute('''Select Sentiment,Count(*) from users group by Sentiment''').fetchall()
    sent_topic = []
    sent_count1 = []
    for tx in sent_count:
        sent_topic.append(tx[0])
        sent_count1.append(tx[1])
    for x in query2:
        clean_text_list.append(x[0])
        if len(x)>1:
            sent_list.append(x[1])

    if len(clean_text_list)==0:
        string="No results found."
        return render_template("index_news.html",string=string,topic_list=topic_list,result=result,source_result=source_result,sent_topic=sent_topic,
                           source_list=source_list,bank_list=bank_list,bank_result=bank_result,start_date=start_date,end_date=end_date)
    list_words = fetch_sentiment_using_vader(clean_text_list)
    stopwords = stopwords_for_wordcount(clean_text_list)
    count_vectorizer = CountVectorizer(stop_words=stopwords[0])
    fig_pos = plot_words(list_words[0], list_words[2], "positive")
    fig_neg = plot_words(list_words[1], list_words[2], "negative")
    create_wordcloud(stopwords)
    fig_tri = bigram_or_trigram(clean_text_list, stopwords, "bigram")

    images_list = os.listdir(os.path.join(app.static_folder, "images"))
    article_text=[]
    title_text=[]
    summary_text=[]
    for row in query3:
        article_text.append(row[1])
        title_text.append(row[0])
        summary_text.append(row[2])
        create_text(row[0],row[1])
        create_summary(row[0],row[2])
    text_list1 = os.listdir(os.path.join(app.static_folder, "text"))
    summary_list1=os.listdir(os.path.join(app.static_folder, "summaries"))
    df1=pd.DataFrame(query,columns=['Sentiment','Date','Publication','Bank','Category','Title','Clean title'])
    df1=(df1.drop(['Sentiment', 'Clean title'], axis=1))
    df2=pd.DataFrame(query3,columns=['Clean title','Raw Article','Summary'])
    df2=(df2.drop(['Clean title'],axis=1))
    frames=[df1,df2]
    df=pd.concat(frames,axis=1)
    print(df)
    df.to_excel('static/table/article_directory.xlsx')
    table_list = os.listdir(os.path.join(app.static_folder, "table"))
    if sent_result == "All":
        fig_pie=count_sent_pie(Counter(sent_list).keys(),Counter(sent_list).values())

        return render_template("index_news.html",topic_list=topic_list,result=result,source_result=source_result,summary=summary_list1,
                           source_list=source_list,sent_topic=sent_topic,bank_list=bank_list,bank_result=bank_result,start_date=start_date,end_date=end_date,
                           max_date=max_date,query=query,fig_pos=fig_pos,string="None",search=search,fig_pie=fig_pie,
                           fig_tri=fig_tri,fig_neg=fig_neg,img1=images_list,text1=text_list1,table_excel=table_list)
    else:

        return render_template("index_news.html",topic_list=topic_list,result=result,source_result=source_result,summary=summary_list1,
                           source_list=source_list,sent_topic=sent_topic,bank_list=bank_list,bank_result=bank_result,start_date=start_date,end_date=end_date,
                           max_date=max_date,query=query,fig_pos=fig_pos,string="None",search=search,sent_result=sent_result,
                           fig_tri=fig_tri,fig_neg=fig_neg,img1=images_list,text1=text_list1,table_excel=table_list)
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
    news_df = pd.read_csv('news_information1.csv')
    news_df.to_sql('users', con=engine)
    sent_list=news_df['Sentiment'].unique()
    topic_l = engine.execute('''Select distinct Topic from users''').fetchall()
    topic_list = []
    for tr in topic_l:
        topic_list.append(tr[0])

    source_l=engine.execute('''Select distinct source from users''').fetchall()
    source_list = []
    for tr in source_l:
        source_list.append(tr[0])
    bank_l = engine.execute('''Select distinct bank from users''').fetchall()
    bank_list = []
    for tr in bank_l:
        bank_list.append(tr[0])
    end_date=engine.execute('''Select max(date) from users''').fetchall()
    for x in end_date:
        max_date=x[0]

    sent_count = engine.execute('''Select Sentiment,Count(*) from users group by Sentiment''').fetchall()
    sent_topic = []
    sent_count1 = []
    for tx in sent_count:
        sent_topic.append(tx[0])
        sent_count1.append(tx[1])
    query = engine.execute('''Select date,source,bank,Topic,Headline,clean_title from users where "Raw Article"
     like ('%' || ? || '%')''',(str(result),) ).fetchall()

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
        return render_template("search_news.html", string=string, topic_list=topic_list, result=result,sent_topic=sent_topic,
                            source_list=source_list, bank_list=bank_list, end_date=end_date,max_date=max_date)
    list_words = fetch_sentiment_using_vader(clean_text_list)
    stopwords = stopwords_for_wordcount(clean_text_list)
    count_vectorizer = CountVectorizer(stop_words=stopwords[0])
    fig_pos = plot_words(list_words[0], list_words[2], "positive")
    fig_neg = plot_words(list_words[1], list_words[2], "negative")
    create_wordcloud(stopwords)
    fig_tri = bigram_or_trigram(clean_text_list, stopwords, "bigram")
    images_list = os.listdir(os.path.join(app.static_folder, "images"))
    article_text = []
    title_text = []
    summary_text = []
    for row in query3:
        article_text.append(row[1])
        title_text.append(row[0])
        summary_text.append(row[2])
        create_text(row[0], row[1])
        create_summary(row[0], row[2])
    df1 = pd.DataFrame(query, columns=[ 'Date', 'Publication', 'Bank', 'Category', 'Title', 'Clean title'])
    df1 = (df1.drop([ 'Clean title'], axis=1))
    df2 = pd.DataFrame(query3, columns=['Clean title', 'Raw Article', 'Summary'])
    df2 = (df2.drop(['Clean title'], axis=1))
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
    print(df)
    df.to_excel('static/table/article_directory.xlsx')
    table_list = os.listdir(os.path.join(app.static_folder, "table"))

    text_list1 = os.listdir(os.path.join(app.static_folder, "text"))
    summary_list1 = os.listdir(os.path.join(app.static_folder, "summaries"))
    fig_pie=count_sent_pie(Counter(sent_list).keys(),Counter(sent_list).values())
    return render_template("search_news.html", topic_list=topic_list, result=result,
                           summary=summary_list1,fig_pie=fig_pie,sent_topic=sent_topic,
                           source_list=source_list, bank_list=bank_list,table_excel=table_list,
                           max_date=max_date, query=query, fig_pos=fig_pos, string="None", search=search,
                           fig_tri=fig_tri, fig_neg=fig_neg, img1=images_list, text1=text_list1)

import time
start = time.process_time()
# your code here
print(time.process_time() - start)
if __name__ == "__main__":
    app.run(debug=True)