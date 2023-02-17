
import os.path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
from pprint import pprint
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim



data_all = pd.read_csv("jindata.csv")


def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern,'',sentence)
    new_sentence = ''.join(line.split())
    return new_sentence
train_text = [clear_character(data_all) for data_all in data_all['review']]


jieba.load_userdict("jiebaDci.txt")

train_seg_text = [jieba.lcut(s) for s in train_text]

stop_words_path = "jinstop.txt"
def get_stop_words():
    return set([item.strip() for item in open(stop_words_path,'r').readlines()])

stopwords = get_stop_words()


def drop_stopwords(line):
    line_clear = []
    for word in line:
        if word in stopwords:
            continue
        line_clear.append(word)
    return line_clear
train_st_text = [drop_stopwords(s) for s in train_seg_text]
data_all['review_st'] = train_st_text


def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear
train_fine_text = [is_fine_word(s,min_length=2) for s in train_st_text]
data_all['review_fine'] = train_fine_text


bigram = gensim.models.Phrases(train_fine_text,min_count=5,threshold=5)
trigram = gensim.models.Phrases(bigram[train_fine_text],threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigram(texts):
    return [trigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)


id2word = corpora.Dictionary(train_st_text)     #create dictionary
texts = train_st_text                          #create corpus
corpus = [id2word.doc2bow(text) for text in texts]    #term document frequency

lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=9,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=1600,
                                           passes=20,
                                           eta='auto',
                                           alpha='auto',
                                           per_word_topics=True)
print(lda_model.alpha)
print(lda_model.eta)
pprint(lda_model.print_topics(num_topics=9,num_words=30))

doc_topic = lda_model.get_document_topics(bow=corpus,minimum_probability=0)
topic_score = pd.DataFrame(doc_topic,columns=["Topic {}".format(i) for i in range(0, 9)])

result = data_all.join(topic_score)
result.to_csv('data_result.csv',encoding="utf-8")


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution  for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 9)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)
df_topic_sents_keywords = data_all.join(df)

df_topic_sents_keywords.to_csv('df_topic_sents_keywords.csv',encoding="utf-8")


d = pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
pyLDAvis.show(d)



