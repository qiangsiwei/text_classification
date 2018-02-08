# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Sequential
from keras.layers import *

class TextClassifierCLSTM1(TextClassifierBase):

	def __init__(self):
		super(TextClassifierCLSTM1,self).__init__()

	def init_model(self):
		model = Sequential()
		model.add(Embedding(len(self.vocab)+1,256,input_length=self.para['maxlen']))
		model.add(Convolution1D(256,3,padding='same',strides=1))
		model.add(Activation('relu'))
		model.add(MaxPool1D(pool_size=2))
		model.add(GRU(256,dropout=0.2,recurrent_dropout=0.1,return_sequences=True))
		model.add(GRU(256,dropout=0.2,recurrent_dropout=0.1))
		model.add(Dense(self.labels,activation='softmax'))
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'x_sqs',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'x_sqs',**kwargs)

if __name__ == '__main__':
	pass
