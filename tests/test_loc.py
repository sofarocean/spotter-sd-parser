#!/usr/bin/env python3

import os
import shutil
import tempfile
import time
import unittest

from math import ceil
from sd_file_parser import parseLocationFiles

class LocationParsingTest(unittest.TestCase):

    def testBasicParserRun(self):
        parseLocationFiles( inputFileName = self.inputfn, kind='LOC', outputFileName=self.outputfn )
        self.assertTrue( os.path.exists( self.outputfn ) ) 
        self.assertFalse( os.path.exists( 'displacement.csv' ) )
    
    def testNoOutputPathParserRun(self):
        """
        show that parseLocationFiles() can't take an output path argument
            ...though its partner parseSpectr[...] can...
            ...probably because it only generates one file whereas the spectral 
               equivalent can output many...
        """
        with self.assertRaises(TypeError) as cm:
            parseLocationFiles( inputFileName = self.inputfn, outputPath = self.outputpath)

    def setUp(self):
        """
        prepare for running the parser
        """
        self.inputfn = 'example_data/2021-01-15/0235_LOC.CSV'        
        # self.outputfn = f"spctest_{ceil(time.time()):x}.csv"
        self.outputfn = 'location.csv'
        self.outputpath = tempfile.mkdtemp()
        if not self.outputpath:
            raise ValueError('problem creating temporary output directory')
        
    def tearDown(self):
        """
        delete temporary output file
        """
        if os.path.exists( self.outputfn ):
            print(f"removing {self.outputfn}")
            os.remove(self.outputfn)
        if os.path.exists(self.outputpath):
            print(f"cleanup: deleting {self.outputpath}")
            shutil.rmtree(self.outputpath)
        
if __name__ == '__main__':
    unittest.main()
    
