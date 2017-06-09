#!/usr/bin/env python
import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from keras.optimizers import Adamax
from keras.constraints import maxnorm
from itertools import combinations



seed = 7
np.random.seed(seed)

StartTime = time.time()

def create_model(input_dim=7):
	model = Sequential()
	model.add(Dense(20, input_dim=input_dim, kernel_initializer='he_uniform', activation='relu', kernel_constraint=maxnorm(5)))
	model.add(Dropout(0.3))
	model.add(Dense(3, kernel_initializer='he_uniform', activation='sigmoid'))
	optimizer = Adamax(lr=0.204)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def create_features(data, degree=3, hash=hash):
	new_data = []
	m,n = data.shape
	for idx in combinations(range(n), degree):
		new_data.append([hash(tuple(v)) for v in data[:,idx]])
	return np.array(new_data).T


def main():

	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test.csv')
	event_type = pd.read_csv('data/event_type.csv')
	log_feature = pd.read_csv('data/log_feature.csv')
	resource_type = pd.read_csv('data/resource_type.csv')
	severity_type = pd.read_csv('data/severity_type.csv')

	X = train.copy()

	print("Removing non-numerics")
	for idx in range(len(train)):
		temp = str(train.iloc[[idx]]['location'])
		temp = temp.split('location')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		X.loc[idx, 'location'] = temp
	X['location'] = X['location'].astype('float32')

	for idx in range(len(test)):
		temp = str(test.iloc[[idx]]['location'])
		temp = temp.split('location')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		test.loc[idx, 'location'] = temp
	test['location'] = test['location'].astype('float32')

	for idx in range(len(event_type)):
		temp = str(event_type.iloc[[idx]]['event_type'])
		temp = temp.split('event_type')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		event_type.loc[idx, 'event_type'] = temp
	event_type['event_type'] = event_type['event_type'].astype('float32')

	for idx in range(len(log_feature)):
		temp = str(log_feature.iloc[[idx]]['log_feature'])
		temp = temp.split('feature')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		log_feature.loc[idx, 'log_feature'] = temp
	log_feature['log_feature'] = log_feature['log_feature'].astype('float32')
	log_feature['volume'] = log_feature['volume'].astype('float32')

	for idx in range(len(resource_type)):
		temp = str(resource_type.iloc[[idx]]['resource_type'])
		temp = temp.split('resource_type')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		resource_type.loc[idx, 'resource_type'] = temp
	resource_type['resource_type'] = resource_type['resource_type'].astype('float32')

	for idx in range(len(severity_type)):
		temp = str(severity_type.iloc[[idx]]['severity_type'])
		temp = temp.split('severity_type')[1]
		temp = ''.join(x for x in temp if x.isdigit())
		severity_type.loc[idx, 'severity_type'] = temp
	severity_type['severity_type'] = severity_type['severity_type'].astype('float32')


	print("Merging train dataset")
	X = pd.merge(X, event_type, how='left', on='id')
	X = pd.merge(X, log_feature, how='left', on='id')
	X = pd.merge(X, resource_type, how='left', on='id')
	X = pd.merge(X, severity_type, how='left', on='id')
	X['id'] = X['id'].astype('float32')
	X.drop_duplicates(subset=['id'], inplace=True, keep='first')
	Y = X['fault_severity']
	del X['fault_severity']

	X = X.values	
	X_2 = create_features(X, degree=2)
	X_3 = create_features(X, degree=3)
	X = np.hstack((X, X_2, X_3))
	X_dim = X.shape[1]
	dummy_y = np_utils.to_categorical(Y)



	print("Merging test dataset")
	test = pd.merge(test, event_type, how='left', on='id')
	test = pd.merge(test, log_feature, how='left', on='id')
	test = pd.merge(test, resource_type, how='left', on='id')
	test = pd.merge(test, severity_type, how='left', on='id')
	test.drop_duplicates(subset=['id'], inplace=True, keep='first')
	ID_vals = test['id']

	test = test.values
	test_2 = create_features(test, degree=2)
	test_3 = create_features(test, degree=3)
	test = np.hstack((test, test_2, test_3))
	test_dim = test.shape[1]


	print("Transforming with StandardScaler")
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	test = scaler.transform(test)


	# Principal Component Analysis
	ncomponents = 35
	pca = PCA(n_components=ncomponents)
	feature_fit = pca.fit(X)
	X = feature_fit.transform(X)
	test = feature_fit.transform(test)
	X_dim = ncomponents

	estimator = KerasClassifier(build_fn=create_model, input_dim=X_dim, batch_size=80, epochs=10)

	print("Starting KFold cross validation")
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X, dummy_y, cv=kfold, scoring='neg_log_loss')
	print("Baseline: %.2f (%.2f%%)" %(results.mean(), results.std()*100))



	estimator.fit(X, dummy_y)
	predictions = estimator.predict_proba(test)

	submissions = pd.DataFrame(data = {'id': ID_vals, 'predict_0': predictions[:,0], 'predict_1': predictions[:,1], 'predict_2': predictions[:,2]})
	submissions.to_csv('output_predictions_extrafeats_featselect_pca35_orig_algo_.csv', index=False)

	EndTime = (time.time() - StartTime) / 60.0
	print("Program took %.2f minutes" % (EndTime))

if __name__ == "__main__":
	main()