# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import *

class TextClassifierCLSTM2(TextClassifierBase):

	def __init__(self):
		super(TextClassifierCLSTM2,self).__init__()

	def init_model(self):
		main_input = Input(shape=(self.para['maxlen'],),dtype='float64')
		embed = Embedding(len(self.vocab)+1,256,input_length=self.para['maxlen'])(main_input)
		cnn = Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
		cnn = MaxPool1D(pool_size=4)(cnn)
		cnn = Flatten()(cnn)
		cnn = Dense(256)(cnn)
		rnn = Bidirectional(GRU(256,dropout=0.2,recurrent_dropout=0.1))(embed)
		rnn = Dense(256)(rnn)
		con = concatenate([cnn,rnn],axis=-1)
		main_output = Dense(self.labels,activation='softmax')(con)
		model = Model(inputs=main_input,outputs=main_output)
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,'x_sqs',**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,'x_sqs',**kwargs)

if __name__ == '__main__':
	pass
