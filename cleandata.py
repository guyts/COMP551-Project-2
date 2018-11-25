import re
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

def readCSV(filename, arr):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			arr.append(row[1])
	arr.pop(0)

def sanitize(text):
    # this function cleans up reddit texts from HTML characters, punctuations
    # and stopwords, as well as uncapitalizes all words.
    
    # cleaning text from <> and other html chars
    tmp = BeautifulSoup(text)  
    # removing punctuations etc, capitals, and splitting into words
    words = (re.sub("[^a-zA-Z]", " ", tmp.get_text())).lower().split()
    # removing stop words as defined in nltk
    meaningfll = [w for w in words if not w in stop_words]
    lst=[]    
    for j in meaningfll:
        tmp3 = lemmatizer.lemmatize(j,pos='v')
        lst.append(tmp3)
    cleanTxt = " ".join(lst)
    return cleanTxt

print "Enter the name of input file\n"
input_file = raw_input()
print "Enter the name of output file\n"
output_file = raw_input()
print "Enter the name of result file\n"
result_file = raw_input()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

train_input = []
readCSV(input_file, train_input)

train_output = []
readCSV(output_file, train_output)

for i in xrange(len(train_input)):
    train_input[i] = sanitize(train_input[i])

f = open(result_file, 'w')
for i in xrange(len(train_input)):
    f.write(str(i) + "," + train_input[i] + "," + train_output[i] + "\n")