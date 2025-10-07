#!/usr/bin/env python3 -m unittest -v

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
    def testCatSst(self):
        for suffix in self.suffixes:
            # just SST right now
            outputFileName = os.path.join(self.output_path, self.outFiles[suffix] + '.csv')
            result = cat(path=self.input_path, Suffix=suffix)

    def setUp(self):
        """
        prepare for running the parser
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.input_path = os.path.join(project_root, 'example_data', '2021-01-15')
        inputfn = '0235_SST.CSV'
        inputfqfn = os.path.join(self.input_path, inputfn)

        self.output_path = os.path.join(self.input_path, 'processed')

        # borrowed from parser script main()
        # self.suffixes = ['FLT','SPC','SYS','LOC','GPS','SST']
        self.suffixes = ['SST']
        self.outFiles    = {'FLT':'displacement','SPC':'spectra','SYS':'system',
                       'LOC':'location','GPS':'gps','SST':'sst'}
        if not self.output_path:
            raise ValueError('problem creating temporary output directory')

        
    def tearDown(self):
        """
        delete temporary output file(s)
        """
        # if os.path.exists( self.outputfn ):
        #     print(f"removing {self.outputfn}")
        #     os.remove(self.outputfn)
        # if os.path.exists(self.outputpath):
        #     print(f"cleanup: deleting {self.outputpath}")
        #     shutil.rmtree(self.outputpath)

        
if __name__ == '__main__':
    unittest.main()
    
