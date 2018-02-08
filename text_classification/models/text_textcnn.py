# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import *

class TextClassifierTextCNN(TextClassifierBase):

	def __init__(self):
		super(TextClassifierTextCNN,self).__init__()

	def init_model(self):
		main_input = Input(shape=(self.para['maxlen'],),dtype='float64')
		embed = Embedding(len(self.vocab)+1,256,input_length=self.para['maxlen'])(main_input)
		cnn1 = Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
		cnn1 = MaxPool1D(pool_size=4)(cnn1)
		cnn2 = Convolution1D(256,4,padding='same',strides=1,activation='relu')(embed)
		cnn2 = MaxPool1D(pool_size=4)(cnn2)
		cnn3 = Convolution1D(256,5,padding='same',strides=1,activation='relu')(embed)
		cnn3 = MaxPool1D(pool_size=4)(cnn3)
		cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)
		flat = Flatten()(cnn)
		drop = Dropout(0.2)(flat)
		main_output = Dense(self.labels,activation='softmax')(drop)
		model = Model(inputs=main_input,outputs = main_output)
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'x_sqs',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'x_sqs',**kwargs)

if __name__ == '__main__':
	pass
