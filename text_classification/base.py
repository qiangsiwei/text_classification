# -*- coding: utf-8 -*-

import os, pickle, numpy as np
from abc import abstractmethod
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from layers import AttLayer

class DataCache(object):
	def __init__(self):
		self.mode = ''
		self.data = {}

	def __format_name__(self,name):
		return '{0}_{1}'.format(self.mode,name)

	def __setitem__(self,name,value):
		name = self.__format_name__(name)
		self.data[name] = value

	def __getitem__(self,name):
		name = self.__format_name__(name)
		return self.data.get(name)

	def __contains__(self,name):
		name = self.__format_name__(name)
		return name in self.data

	def set_mode(self,mode):
		self.mode = mode

class TextClassifierBase(object):
	def __init__(self,tokenizer=None,maxlen=50):
		self.data = DataCache()
		self.para = DataCache()
		self.model = None
		self.para['maxlen'] = maxlen
		self.para['tokenizer'] = tokenizer
		if not self.para['tokenizer']:
			self.para['tokenizer'] = Tokenizer(\
				filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
		self.custom_layers = {'AttLayer':AttLayer}

	def __proc__x(self, x):
		self.data['x_ids'] = \
			np.array(self.para['tokenizer'].texts_to_sequences(x))
		self.data['x_sqs'] = \
			pad_sequences(self.data['x_ids'],maxlen=self.para['maxlen'])
		self.data['x_mtx'] = \
			self.para['tokenizer'].sequences_to_matrix(self.data['x_ids'],mode='binary')

	def init_fit_data(self, x, y, test_size=0):
		self.para['tokenizer'].fit_on_texts(x)
		self.vocab = self.para['tokenizer'].word_index
		self.__proc__x(x)
		self.data['y'] = to_categorical(y)
		self.labels = self.data['y'].shape[-1]

	def proc_fit_data(self, **kwargs):
		pass

	def init_pred_data(self, x):
		self.vocab = self.para['tokenizer'].word_index
		self.__proc__x(x)

	def proc_pred_data(self, **kwargs):
		pass

	@abstractmethod
	def init_model(self):
		pass

	def __get_x__(self,name):
		assert isinstance(name,str)
		assert name in self.data
		return self.data[name]

	def __fit__(self, x, y, inputs, **kwargs):
		self.mode = 'fit'
		self.init_fit_data(x,y)
		self.proc_fit_data(kwargs=kwargs)
		self.model = self.init_model()
		x = self.__get_x__(inputs) if isinstance(inputs,str) else\
			map(self.__get_x__,inputs)
		self.model.fit(
			x = x,
			y = self.data['y'],
			epochs=kwargs.get('epochs',10),
			validation_split=kwargs.get('validation_split',0))

	@abstractmethod
	def fit(self, x, y, **kwargs):
		pass

	def __predict__(self, x, inputs, **kwargs):
		self.mode = 'predict'
		assert self.model != None
		self.init_pred_data(x)
		self.proc_pred_data(kwargs=kwargs)
		x = self.__get_x__(inputs) if isinstance(inputs,str) else\
			map(self.__get_x__,inputs)
		y = self.model.predict(x)
		return y.argmax(axis=1)

	@abstractmethod
	def predict(self, x, **kwargs):
		pass

	def __get_filenames__(self, dirn):
		return os.path.join(dirn,'tokenizer.pkl'),\
			   os.path.join(dirn,'model.json'),\
			   os.path.join(dirn,'param.h5')

	def dump_model(self, dirn):
		if not os.path.isdir(dirn):
			os.mkdir(dirn)
		assert self.model != None # more strict
		ft, fm, fp = self.__get_filenames__(dirn)
		with open(ft,'wb') as out:
			pickle.dump(self.para,out)
		with open(fm,'w') as out: 
			out.write(self.model.to_json())
		self.model.save_weights(fp,overwrite=True)

	def load_model(self, dirn):
		assert os.path.isdir(dirn)
		ft, fm, fp = self.__get_filenames__(dirn)
		assert all(os.path.isfile(fn) for fn in (ft,fm,fp))
		self.para = pickle.load(open(ft,'rb'))
		self.model = model_from_json(open(fm).read(),self.custom_layers)
		self.model.load_weights(fp)

if __name__ == '__main__':
	pass
