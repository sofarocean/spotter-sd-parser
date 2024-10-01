#!/usr/bin/env python3 -m unittest

"""
test location file parsing
"""

import os
import shutil
import sys
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

    def testOutputLocationIsValidFloat(self):
        # test against bug where V3 LOC files are misinterpreted as not being floats
        parseLocationFiles( inputFileName = r'example_data/2024-09-23/0002_LOC.csv', kind='LOC', outputFileName=self.outputfn )
        self.assertTrue( os.path.exists( self.outputfn ) ) 
        # print(f"wrote file to {self.outputpath}...?", file=sys.stderr)
        with open(self.outputfn, 'r') as csvf:
            # bad: 2024,9,23,14,54,24,0,  37.00000000,-122.00000000
            # skip header
            _ = csvf.readline()
            firstline = csvf.readline()
        self.assertFalse(firstline.rstrip().endswith(',  37.00000000,-122.00000000'))

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
        if os.path.exists("displacement.csv"):
            os.remove("displacement.csv")
        self.inputfn = 'example_data/2021-01-15/0235_LOC.CSV'        
        self.outputfn = f"location_{ceil(time.time()):x}.csv"
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
    
