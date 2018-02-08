# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Sequential
from keras.layers import *

class TextClassifierMLP(TextClassifierBase):

	def __init__(self):
		super(TextClassifierMLP,self).__init__()

	def init_model(self):
		model = Sequential()
		model.add(Dense(512,input_shape=(len(self.vocab)+1,),activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.labels,activation='softmax'))
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'x_mtx',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'x_mtx',**kwargs)

if __name__ == '__main__':
	pass
