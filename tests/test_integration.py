#!/usr/bin/env python3 -m unittest

"""
test epoch time conversion routines
"""

import os
import shutil
import tempfile
import time
import unittest

from sd_file_parser import parseLocationFiles
from sd_file_parser import parseSpectralFiles
from sd_file_parser import main
import sd_file_parser
import sd_csv_to_netcdf
import xarray as xr

class IntegrationTest(unittest.TestCase):
    def test_sd_file_parser(self):
        """
        TODO
        """
        sd_file_parser.main(path=self.inpath,
                            outpath=self.outputpath_01,
                            outputFileType='CSV',
                            spectra='all',
                            suffixes=None,
                            parsing=None,
                            lfFilter=False,
                            bulkParameters=True)

        ds1 = sd_csv_to_netcdf.main(path=self.outputpath_01, 
                                    outpath=self.outputpath_01)
      
        out_file = os.path.join(self.outputpath_01, 'spot.nc')
        self.assertTrue(os.path.exists(out_file))
        ds2 = xr.open_dataset(out_file)
        
        #TODO assert the Dataset contains all the expected variables, etc.

    def setUp(self):
        self.inpath = os.path.join('example_data','2021-01-15')
        self.outputpath_01 = tempfile.mkdtemp()
        if not self.outputpath_01:
            raise ValueError('problem creating temporary output directory')


if __name__ == '__main__':
    unittest.main()
