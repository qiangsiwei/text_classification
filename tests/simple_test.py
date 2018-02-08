# -*- coding: utf-8 -*-

import unittest

class SimpleTestCase(unittest.TestCase):

	def test_import(self):
		import text_classification
		return True

if __name__ == "__main__":
	suite = unittest.TestSuite()
	test_cases = ['test_import']
	for test_case in test_cases:
		suite.addTest(SimpleTestCase(test_case))
	unittest.TextTestRunner(verbosity=2).run(suite)
