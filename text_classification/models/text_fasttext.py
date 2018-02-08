# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class TextClassifierFastText(TextClassifierBase):

	def __init__(self):
		super(TextClassifierFastText,self).__init__()

	def init_model(self):
		model = Sequential()
		model.add(Embedding(self.para['fasttext_dim'],256,input_length=self.para['fasttext_maxlen']))
		model.add(GlobalAveragePooling1D())
		model.add(Dense(self.labels,activation='softmax'))
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def __add_ngrams__(self, seqs):
		gram_n = self.para['gram_n']
		token_to_idx = self.para['token_to_idx']
		for inputs in seqs:
			nlist = inputs[:]
			for i in range(len(nlist)-gram_n+1):
				for ngram_value in range(2,gram_n+1):
					ngram = tuple(nlist[i:i+gram_n])
					if ngram in token_to_idx:
						nlist.append(token_to_idx[ngram])
			yield nlist

	def __proc_data__(self):
		self.data['fasttext_x_sqs'] = pad_sequences(\
			list(self.__add_ngrams__(self.data['x_ids'])),\
			maxlen=self.para['fasttext_maxlen'])

	def proc_fit_data(self, **kwargs):
		self.para['gram_n'] = max(kwargs.get('gram_n') or 2,2)
		self.para['fasttext_maxlen'] = kwargs.get('maxlen') or 2*self.para['maxlen']
		def create_ngrams(inputs, gram_n):
			return set(zip(*[inputs[i:] for i in range(gram_n)]))
		ngrams = set()
		for inputs in self.data['x_sqs']:
			for i in range(2,self.para['gram_n']+1):
				ngrams.update(create_ngrams(inputs,gram_n=i))
		start_idx = len(self.vocab)+2
		token_to_idx = {token:idx+start_idx for idx,token in enumerate(ngrams)}
		idx_to_token = {idx:token for token,idx in token_to_idx.iteritems()}
		self.para['token_to_idx'] = token_to_idx
		self.para['fasttext_dim'] = np.max(list(idx_to_token.keys()))+1
		self.__proc_data__()

	def proc_pred_data(self, **kwargs):
		self.__proc_data__()

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'fasttext_x_sqs',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'fasttext_x_sqs',**kwargs)

	def dump_model(self, dirn):
		super(TextClassifierFastText,self).dump_model(dirn)

	def load_model(self, dirn):
		super(TextClassifierFastText,self).load_model(dirn)

if __name__ == '__main__':
	pass
