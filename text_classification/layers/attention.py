# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, constraints, regularizers

class AttLayer(Layer):
	def __init__(self,init='glorot_uniform',kernel_regularizer=None,bias_regularizer=None,\
		kernel_constraint=None,bias_constraint=None,**kwargs):
		self.supports_masking = True
		self.init = initializers.get(init)
		self.kernel_initializer = initializers.get(init)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.bias_constraint = constraints.get(bias_constraint)
		super(AttLayer, self).__init__(** kwargs)
	def build(self, input_shape):
		assert len(input_shape)==3
		self.W = self.add_weight((input_shape[-1],1),initializer=self.kernel_initializer,\
			name='{}_W'.format(self.name),regularizer=self.kernel_regularizer,constraint=self.kernel_constraint)
		self.b = self.add_weight((input_shape[1],),initializer='zero',\
			name='{}_b'.format(self.name),regularizer=self.bias_regularizer,constraint=self.bias_constraint)
		self.u = self.add_weight((input_shape[1],),initializer=self.kernel_initializer,\
			name='{}_u'.format(self.name),regularizer=self.kernel_regularizer,constraint=self.kernel_constraint)
		self.built = True
	def compute_mask(self, input, input_mask=None):
		return None
	def call(self, x, mask=None):
		ait = K.exp(K.tanh(K.squeeze(K.dot(x,self.W),-1)+self.b)*self.u)
		if mask is not None: mask = K.cast(mask,K.floatx()); ait = mask*ait
		ait /= K.cast(K.sum(ait,axis=1,keepdims=True)+K.epsilon(),K.floatx())
		ait = K.expand_dims(ait)
		return K.sum(x*ait,axis=1)
	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[-1])

if __name__ == '__main__':
	pass
