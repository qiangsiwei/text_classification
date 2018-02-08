# -*- coding: utf-8 -*-

import fileinput

def get_data():
	for i in range(3):
		for l in fileinput.input('data/{}.txt'.format(i)):
			yield l.decode('utf-8'), i

if __name__ == '__main__':
	pass
	