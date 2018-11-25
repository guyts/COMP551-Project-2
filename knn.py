import math
import collections, re
import random
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split

DOWNRATIO = 10
K = 150
MAXFEATURES = 1000

# Read a csv file into a list
def readCSV(filename, arr):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			arr.append(row)

# Get similarity
def getEuclideanDist(a, b, length):
	dist = 0
	for i in xrange(length):
		dist += (a[i]-b[i])**2
	return dist

# Get the k nearest neighbours
def getNeighbors(train_x, testPt, k):
	dists = []
	# Get all distances of every point to target point
	for i in xrange(len(train_x)):
		dist = getEuclideanDist(train_x[i], testPt, len(train_x[i]))
		dists.append([dist, i])

	# Select the k nearest ones
	dists.sort()

	neighbours = []

	for i in xrange(k):
		neighbours.append(dists[i])

	return neighbours

# Return the majority category among the k neighbours
def getMajority(neighbours, train_y):
	# Compute the frequency of each category
	frequencies = {}
	for i in xrange(len(neighbours)):
		category = train_y[neighbours[i][1]]
		if category not in frequencies:
			frequencies[category] = 1
		else:
			frequencies[category] += 1

	output = ""
	maxCount = 0

	# Find the most recurrent category
	for category in frequencies:
		if frequencies[category] > maxCount:
			output = category
			maxCount = frequencies[category]

	return output

# Return the ratio of correct predictions
def getError(test_data, test_prediction):
	n = len(test_data)
	correct = 0
	for i in xrange(len(test_data)):
		# print test_data[i], test_prediction[i]
		if test_data[i] == test_prediction[i]:
			correct += 1

	return float(correct)/float(n)

# Get file names
print "Enter the name of training file\n"
train_file = raw_input()
print "Enter the name of testing file\n"
test_file = raw_input()
print "Enter the name of result file\n"
result_file = raw_input()

# Get train input and output from csv
data = []
readCSV(train_file, data)

texts = []
y = []

texts = [data[i][1] for i in xrange(len(data)/DOWNRATIO)]
y = [data[i][2] for i in xrange(len(data)/DOWNRATIO)]

# Get bag of words
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = MAXFEATURES)

feats = vectorizer.fit_transform(texts)
features = feats.toarray()

# Apply TF-Idf
tf_transformer = TfidfTransformer(use_idf=True).fit(feats)
X = tf_transformer.transform(feats)
X = X.toarray().tolist()

# Split train, test data
X_train, X_test, y_train, y_test = train_test_split(X, y)
test_prediction = []

# Predict
for i in xrange(len(X_test)):
	if i%100 == 0: print i
	neighbours = getNeighbors(X_train, X_test[i], K)
	test_prediction.append(getMajority(neighbours, y_train))

# Get error
print getError(y_test, test_prediction)

###################### Prediction ###########################

# Add additional data
data_test = []
readCSV(test_file, data_test)
texts = texts + [row[1] for row in data_test]

# Reevaluate features
feats = vectorizer.fit_transform(texts)
features = feats.toarray()

# Apply Tf-Idf
tf_transformer = TfidfTransformer(use_idf=True).fit(feats)
X = tf_transformer.transform(feats)
X = X.toarray().tolist()

# Split data
X_train = X[0:len(data)/DOWNRATIO]
X_test = X[len(data)/DOWNRATIO:]

# Predict data
test_prediction = []

for i in xrange(len(X_test)):
	if i%100 == 0: print i
	neighbours = getNeighbors(X_train, X_test[i], K)
	test_prediction.append(getMajority(neighbours, y))

# Write to csv
f = open(result_file, 'w')
f.write("id,category\n")
for i in xrange(len(test_prediction)):
	f.write(str(i) + "," + test_prediction[i] + "\n")




