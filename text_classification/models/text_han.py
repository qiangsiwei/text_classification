# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from ..layers import AttLayer
from keras.models import Model
from keras.layers import *

class TextClassifierHAN(TextClassifierBase):

	def __init__(self):
		super(TextClassifierHAN,self).__init__()

	def init_model(self):
		inputs = Input(shape=(self.para['maxlen'],),dtype='float64')
		embed = Embedding(len(self.vocab)+1,256,input_length=self.para['maxlen'])(inputs)
		gru = Bidirectional(GRU(128,dropout=0.2,recurrent_dropout=0.1,return_sequences=True))(embed)
		attention = AttLayer()(gru)
		output = Dense(self.labels,activation='softmax')(attention)
		model = Model(inputs,output)
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'x_sqs',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'x_sqs',**kwargs)

if __name__ == '__main__':
	pass
