import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from time import gmtime, strftime

def load_files(max_features):
	data = []
	f = open('processed_text_train.txt', 'r')
	lines = f.readlines()

	for l in lines:
		line_list = eval(l)
		line = " ".join(line_list)
		data.append(line)
	f.close()

	TF = TfidfVectorizer(max_features = max_features).fit_transform(data)

	return TF

def to_pickle (file_name, object):
    fileObject = open(file_name,'wb')
    pickle.dump(object, fileObject)
    fileObject.close()


if __name__ == '__main__':
	log = open("log_X_validation.txt", 'w')
	log.write ("Begin Program: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
	log.flush()
	training_data = load_files(20000)
	log.write ("Done Processing: " + strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
	log.flush()
	output_df = pd.read_csv("train_output.csv")
	category = output_df.ix[:, 1].as_matrix()

	param_grid = {'C': stats.expon(scale=100)}

	random_search = RandomizedSearchCV(LinearSVC(), param_distributions=param_grid,
	                               n_iter=200, n_jobs=-1, random_state=50)
	log.write ("Start Searching: " + strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
	log.flush()
	random_search.fit(training_data, category)
	log.write ("Done Searching: " + strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
	log.flush()
	
	to_pickle("result.p", random_search)
	log.write ("Finished Pickle: " + strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
	log.flush()












