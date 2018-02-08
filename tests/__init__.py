# -*- coding: utf-8 -*-

import unittest

def get_tests():
    from simple_test import SimpleTestCase
    suite = unittest.TestLoader().loadTestsFromTestCase(SimpleTestCase)
    return unittest.TestSuite([suite])
