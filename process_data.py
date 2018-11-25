from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

stopwords = set(sw.words('english'))
punct = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def tokenize(doc):

    tokens = []
    soup = BeautifulSoup(doc, 'lxml')
    document = soup.get_text()
    for sent in sent_tokenize(document):
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            if token in stopwords:  #Ignore Stopwords
                continue

            if all(char in punct for char in token):    #Ignore Punctuations
                continue

            lemmatized = lemmatize(token, tag)
            tokens.append(lemmatized)
    return tokens

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

    return lemmatizer.lemmatize(token, tag)

f = open('processed_text_test.txt', 'w')
df_train_input = pd.read_csv("test_input.csv")
convos = df_train_input['conversation']
X = np.array([])
numConvos = X.size
for index, c in enumerate(convos):
    f.write("%s\n" % tokenize(c))
f.close()






