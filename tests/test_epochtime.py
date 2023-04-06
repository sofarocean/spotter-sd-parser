#!/usr/bin/env python3

"""
test epoch time conversion routines
"""

import unittest
from sd_file_parser import epochToDateArray

class EpochToDateArrayTest(unittest.TestCase):
    def testArrayInput(self):
        """
        test that the input to the method is an array-like iterable
        """
        with self.assertRaises(TypeError) as cm:
            epochToDateArray(self.epochTimeDecimal1)
        
    def testDecimal(self):
        """
        test against a list of lists
        """
        converted = epochToDateArray(self.epochTimeDecimals)
        # print(converted)
        self.assertSequenceEqual(converted.tolist(), [ [2022, 10, 26, 18, 31, 38, 0] ])
        
    def setUp(self):
        self.epochTimeDecimal1 = 1666809098.00        
        self.epochTimeDecimals = [ self.epochTimeDecimal1 ]
        
if __name__ == '__main__':
    unittest.main()
    
