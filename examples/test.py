# -*- coding: utf-8 -*-

import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('..')
from utils import get_data
from functools import wraps
from text_classification import *
from text_classification.base import TextClassifierBase

def timethis(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		print args[0].__name__
		start = time.time()
		value = func(*args, **kwargs)
		print 'running time:', time.time()-start
		return value
	return wrapper

@timethis
def test_model_train(Classifier, x, y, epochs=10, validation_split=0):
	assert issubclass(Classifier,TextClassifierBase)
	clf = Classifier()
	clf.fit(x,y,epochs=epochs,validation_split=validation_split)
	return clf

@timethis
def test_model_dump(Classifier, x, y, epochs=10, dirn='model'):
	clf = Classifier()
	clf.fit(x,y,epochs=epochs)
	clf.dump_model(dirn)

@timethis
def test_model_load(Classifier, x, y, epochs=10, dirn='model'):
	clf = Classifier()
	clf.load_model(dirn)
	print clf.predict(x)

if __name__ == '__main__':
	validation_split = 0.3
	# Classifier = TextClassifierMLP
	# Classifier = TextClassifierRNN
	# Classifier = TextClassifierCNN
	# Classifier = TextClassifierHAN
	# Classifier = TextClassifierRCNN
	# Classifier = TextClassifierBiRNN
	# Classifier = TextClassifierCLSTM1
	# Classifier = TextClassifierCLSTM2
	# Classifier = TextClassifierTextCNN
	Classifier = TextClassifierFastText

	x,y = zip(*get_data())
	# test_model_train(Classifier,x,y,1,validation_split)
	test_model_dump(Classifier,x,y)
	test_model_load(Classifier,x,y)
