from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import string

def load_files():
	combined = []
	f = open('processed_text_train.txt', 'r')
	lines = f.readlines()
	num_train_convos = len(lines)

	for l in lines:
		line_list = eval(l)
		line = " ".join(line_list)
		combined.append(line)
	f.close()

	f = open('processed_text_test.txt', 'r')
	lines = f.readlines()
	num_test_convos = len(lines)
	for l in lines:
		line_list = eval(l)
		line = " ".join(line_list)
		combined.append(line)
	f.close()

	combined_TF = TfidfVectorizer().fit_transform(combined)

	training_data = combined_TF[0:num_train_convos]
	real_testing_X = combined_TF[num_train_convos:]

	return training_data, real_testing_X

if __name__ == '__main__':
	training_data, real_testing_X = load_files()
	output_df = pd.read_csv("train_output.csv")
	category = output_df.ix[:, 1].as_matrix()

	clf = LinearSVC(C=0.97).fit(training_data, category)
	prediction_result = dict()
	for index, i in enumerate(real_testing_X):
		prediction = clf.predict(i)
		prediction_result[index] = prediction

	df_result = pd.DataFrame(prediction_result).T
	df_result.to_csv('testing_output.csv')





	













