# -*- coding: utf-8 -*-

from ..base import TextClassifierBase
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

class TextClassifierRCNN(TextClassifierBase):

	def __init__(self):
		super(TextClassifierRCNN,self).__init__()

	def init_model(self):
		doc = Input(shape=(None,),dtype='int32')
		lc = Input(shape=(None,),dtype='int32')
		rc = Input(shape=(None,),dtype='int32')
		embedder = Embedding(len(self.vocab)+1,256,input_length=self.para['maxlen'])
		demb, lemb, remb = embedder(doc), embedder(lc), embedder(rc)
		fward = LSTM(256,return_sequences=True)(lemb)
		bward = LSTM(256,return_sequences=True,go_backwards=True)(remb)
		together = concatenate([fward,demb,bward],axis=2)
		semantic = TimeDistributed(Dense(128,activation='tanh'))(together)
		pool_rnn = Lambda(lambda x:K.max(x,axis=1),output_shape=(128,))(semantic)
		output = Dense(self.labels,activation='softmax')(pool_rnn)
		model = Model(inputs=[doc,lc,rc],outputs=output)
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return model

	def __proc_data__(self):
		x_lids = [[len(self.vocab)]+x[:-1]for x in self.data['x_ids'].tolist()]
		x_rids = [x[1:]+[len(self.vocab)] for x in self.data['x_ids'].tolist()]
		self.data['x_lids'] = pad_sequences(x_lids,maxlen=self.para['maxlen'])
		self.data['x_rids'] = pad_sequences(x_rids,maxlen=self.para['maxlen'])

	def proc_fit_data(self, **kwargs):
		self.__proc_data__()

	def proc_pred_data(self, **kwargs):
		self.__proc_data__()

	def fit(self, x, y, **kwargs):
		self.__fit__(x,y,inputs=['x_sqs','x_lids','x_rids'],**kwargs)

	def predict(self, x, **kwargs):
		return self.__predict__(x,['x_sqs','x_lids','x_rids'],**kwargs)

if __name__ == '__main__':
	pass
