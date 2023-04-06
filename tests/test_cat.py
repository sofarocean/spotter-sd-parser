#!/usr/bin/env python3

"""
test the concatenation function of parser
"""

import os
import shutil
import tempfile
import time
import unittest

from math import ceil
from sd_file_parser import cat

class CatTest(unittest.TestCase):

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = 'example_data/2021-01-15/0235_SST.CSV'
        self.outputfn = 'sst.csv'
        
    def tearDown(self):
        """
        delete temporary output file(s)
        """
        # for fn in ['displacement.csv', self.outputfn ]:
        #    if os.path.exists(fn):
        #        print(f"removing {fn}")
        #        os.remove(fn)
        
if __name__ == '__main__':
    unittest.main()
    
