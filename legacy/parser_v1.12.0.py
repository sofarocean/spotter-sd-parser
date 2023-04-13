#!/usr/bin/env python
""" 
Purpose:

    For efficiency, SPOTTER stores wave spectra, mean location and
    displacement data on the SD card across multiple files. In order to 
    access the data for post processing it is convenient to first 
    recombine each data type in a single file. 

    This module contains functions to process SPOTTER output files containing 
    spectra, mean location and displacement information, and concatenate all 
    files pertaining to a specific data type (e.g. displacements) into a single
    comma delimited (CSV) file. For example, all displacement information 
    (contained in ????_FLT.CSV) is combined as

       (input)                                (output) 
    0010_FLT.CSV  -|
    0011_FLT.CSV   |      running script
    0012_FLT.CSV   |           ==== >       displacement.CSV
    ............   |
    000N_FLT.CSV  -|

    and similarly for spectral ( xxxx_SPC.CSV => Szz.csv) and location 
    (xxxx_LOC.CSV => location.csv) files. Further, after all spectral files have
    been combined. Bulk parameters (significant wave height, peak periodm etc.) are calculated from the spectral files,
    and stored seperately in bulkparameters.csv

    NOTE: the original data files will remain unchanged.

Install:

    In order to use this script, python (version 2 or 3) needs to be installed
    on the system (download at: www.python.org). In addition, for functionality 
    the script requires that the following python python modules:

        dependencies: pandas, numpy, scipy
         
    These modules can be installed by invoking the python package manager
    (pip) from the command line. For instance, to install pandas you would
    run the package manager from the command line as:

        pip install pandas

    and similarly for other missing dependencies.

Usage:

    To use the module, simply copy the SPOTTER files and this script into the
    same directory. Subsequently, start a command line terminal, navigate 
    to the directory containing the files and run the python script from the 
    command line using the python interpreter as:

        python sd_file_parser.py

    or any other python interpreter (e.g. ipython, python3 etc.).

    Requesting additional output:

        By default, the script will only produce the variance density spectrum.
        If in addition the directional moments are desired, add the command line
        switch spectra=all, i.e.:

        python spotter.py spectra='all'

        in which case files containing a1,b1,a2,b2 (in separate files) will be
        produced.

Output:

    After completion, the following files will have been created in the working
    directory:

        FILE              :: DESCRIPTION
        ------------------------------------------------------------------------
        Szz.csv           :: Variance density spectra of vertical displacement [meter * meter / Hz]
        Sxx.csv           :: Variance density spectra of eastward displacement [meter * meter / Hz]
        Syy.csv           :: Variance density spectra of northward displacement [meter * meter / Hz]
        Qxz.csv           :: Quad-spectrum between vertical and eastward displacement [meter * meter / Hz]
        Qyz.csv           :: Quad-spectrum between vertical and northward displacement [meter * meter / Hz]
        Cxy.csv           :: Co-spectrum between northward and eastward displacement [meter * meter / Hz]
        a1.csv            :: First order cosine coefficient [ - ]
        b1.csv            :: First order sine coefficient   [ - ]
        a2.csv            :: Second order cosine coefficient  [ - ]
        b2.csv            :: Second order sine coefficient  [ - ]
        location.csv      :: Average location (lowpass filtered instantaneous
                             location) in terms of latitude and longitude
                             (decimal degrees)
        displacement.csv  :: Instantaneous displacement from mean location 
                             along north, east and vertical directions(in meter)
        bulkparameters    :: Bulk wave parameters (Significant wave height, peak peariod, etc.)

     Data is stored as comma delimited file, where each new line corresponds to 
     a new datapoint in time, and the individual columns contain different data
     entries (time, latitude, longitude etc.).

     The spectra files start at the first line with a header line and each
     subsequent line contains the wave spectrum calculated at the indicated time

     HEADER:   year,month,day,hour,min,sec,milisec,dof , 0.0 , f(1) , f(2) , .... , (nf-1) * df
               2017,11   ,10 ,5   ,3  ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)
               2017,11   ,10 ,5   ,33 ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)
                |    |    |   |    |   |   |        |    |    |       |     |
               2017,12   ,20 ,0   ,6  ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)

    The first columns indicate the time (year, month etc.) and dof is the 
    degrees of freedom (dof) used to calculate the spectra. After 
    the degrees of freedom, each subsequent entry corresponds to the variance 
    density at the frequency indicated by the header line (E0 is the energy in
    the mean, E1 at the first frequency f1 etc). The Spotter records
    at an equidistant spectral resolution of df=0.009765625 and there are
    nf=128 spectral entries, given by f(j) = df * j (with 0<=j<128). Frequencies are
    in Hertz, and spectral entries are given in squared meters per Hz (m^2/Hz) or 
    are dimensionless (for the directional moments a1,a2,b1,b2).

    The bulk parameter (bulkparameters.csv) file starts with a header line and subsequent lines contain the bulk
    parameters calculated at the indicated time

    HEADER:    # year , month , day, hour ,min, sec, milisec , Significant Wave Height, Mean Period, Peak Period, Mean Direction, Peak Direction, Mean Spreading, Peak Spreading
               2017,11   ,10 ,5   ,3  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
               2017,11   ,10 ,5   ,33 ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
                |    |    |   |    |   |   |        |     | ,   | , | , |  , |   , |  , |
               2017,12   ,20 ,0   ,6  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr

    For the definitions used to calculate the bulk parameters from the variance density spectra, and a short description
    please refer to:
    https://content.sofarocean.com/hubfs/Spotter%20product%20documentation%20page/wave-parameter-definitions.pdf


Major Updates:

    Author   | Date      | Firmware Version | Script updates
    -----------------------------------------------------------------------
    P.B.Smit | Feb, 2018 | 1.4.2            | firmware SHA verification
    P.B.Smit | May, 2018 | 1.5.1            | Included IIR phase correction
    P.B.Smit | June, 2019| 1.7.0            | Bulk parameter output
    P.B.Smit | Oct, 2019 | 1.8.0            | SST Spotter update

"""

#
# Implementation
#----------------
#


#'SHA <-> version-number' relation
#(note that duel entry for 1.2.5/1.4.2 is due to update glitches)
supportedVersions    = {'1446ABC':'1.2.5','9BEADBE':'1.3.0','2FDC90':'1.4.1',
                        '928D2AE':'1.2.5','3D3CFF5':'1.?.?','A0F6980':'1.?.?',
                        '340b03f':'1.4.2','82755AE':'1.4.2','B218FBD':'1.5.1',
                        'E52AC4D':'1.5.2','1323D38':'1.5.3','73A3A4D0':'1.6.0',
                        '6171C497':'1.6.2','A98A2E52':'1.7.0','A4FAAEBA':'1.7.1',
                        'E7C7CD94':'1.8.0','412432D3':'1.9.0','9F438E3C':'1.9.1',
                        '97A11B27':'1.10.0','2992193B':'1.11.0','2569BD17':'1.11.1',
                        '81C1B398':'1.11.2','93CE0B95':'1.12.0',}

#number assigned to each release, and compatibility number (each release
#    gets a new number, and shares a compatibility number with all releases
#    that can be concatenated in single output files.
#
ordinalVersionNumber = {'1446ABC':(0,0),'9BEADBE':(1,0),'2FDC90':(2,1),
                        '928D2AE':(0,0),'340b03f':(3,1),'82755AE':(3,1),
                        'B218FBD':(4,2),'E52AC4D':(5,2),'1323D38':(6,2),
                        '73A3A4D0':(7,2),'6171C497':(8,2),'A98A2E52':(9,2),
                        'A4FAAEBA':(10,2),'E7C7CD94':(11,2),'412432D3':(12,2),
                        '9F438E3C':(13,2),'97A11B27':(14,2),'2992193B':(15,2),
                        '2569BD17':(16,2),'81C1B398':(17,2),'93CE0B95':(18,2),
                        }
#
# The default version number is used by spectral and location parsing routines
# as default compatibility number if none is given
#
defaultVersion       = 0
defaultIIRWeightType = 0
#
#Phase correction by applying the IIR to the reversed
# signal
#
applyPhaseCorrection                  = True
applyPhaseCorrectionFromVersionNumber = 2
    
import numpy
class Spectrum:
    _parser_files = {'Szz':'Szz.csv', 'a1':'a1.csv', 'b1':'b1.csv','Sxx':'Sxx.csv','Syy':'Syy.csv','Qxz':'Qxz.csv','Qyz':'Qyz.csv'}
    _toDeg = 180. / numpy.pi

    def __init__(self,path,outpath):
        import os

        self._file_available = {'Szz':False, 'a1':False, 'b1':False,}
        self.path = path
        self.outpath = outpath
        self._data = {'Szz':None,'a1':None,'b1':None,'Sxx':None,'Syy':None,'Qxy':None,'Qyz':None}
        self._none = None
        self.time  = None
        for key in self._parser_files:
            #
            # Check which output from the parser is available
            if os.path.isfile(os.path.join(self.path,self._parser_files[key])):

                self._file_available[key] = True

        #Load a header file from the spectral data from the parser to get frequencies
        self._load_header()
        #Load the data from the parser
        self._load_parser_output()

    @property
    def Szz(self):
        return self._data['Szz']

    @property
    def a1(self):
        return self._data['a1']

    @property
    def b1(self):
        return self._data['b1']

    def _Qyzm(self):
        import  numpy
        return numpy.mean( self._data['Qyz'],1 )

    def _Qxzm(self):
        import  numpy
        return numpy.mean( self._data['Qxz'],1 )

    def _Sxxm(self):
        import  numpy
        return numpy.mean( self._data['Sxx'],1 )

    def _Syym(self):
        import  numpy
        return numpy.mean( self._data['Syy'],1 )

    def _Szzm(self):
        import  numpy
        return numpy.mean( self._data['Szz'],1 )

    @property
    def f(self):
        return self._frequencies

    def _load_header(self):
        import numpy
        import os
        for key in self._parser_files:
            if self._file_available[key]:
                with open(os.path.join(self.path,self._parser_files[key]),'r') as file:
                    line = file.readline(  ).split(',')[8:]
                    self._frequencies = numpy.array([float(x) for x in line])
            break
        else:
            raise Exception('No spectral files available - please make sure the script is in the same directory as the sd-card output')


    def _load_parser_output( self ):
        #
        import numpy
        import os
        for key in self._parser_files:
            if self._file_available[key]:
                data = numpy.loadtxt( os.path.join(self.path,self._parser_files[key]) , delimiter=',' )
                self.time = data[ : , 0:8 ]
                self._data[key] = data[:,8:]
                mask = numpy.isnan( self._data[key] )
                self._data[key][mask] = 0.
                self._none = numpy.nan + numpy.zeros( self._data[key].shape )

        for key in self._parser_files:
            if not self._file_available[key]:
                self._data[key] = self._none


    def _moment(self , values ):
        import numpy

        df = numpy.mean(numpy.diff( self.f))
        E = self.Szz * values
        jstart =3
        return numpy.trapz(  E[:,jstart:],self.f[jstart:] ,1  )

    def _weighted_moment(self , values ):
        import numpy

        return self._moment( values ) / self._moment(1.)

    @property
    def a1m(self):
        import  numpy
        if self._file_available['Sxx']:
            return self._Qxzm() / numpy.sqrt( self._Szzm() * ( self._Sxxm() + self._Syym() ))
        else:
            return self._weighted_moment( self.a1 )

    @property
    def b1m(self):
        if self._file_available['Sxx']:
            return self._Qyzm() / numpy.sqrt( self._Szzm() * ( self._Sxxm() + self._Syym() ))
        else:
            return self._weighted_moment( self.b1 )

    def _peak_index(self):
        import numpy
        return numpy.argmax( self.Szz,1 )

    def _direction(self,a1,b1):
        import numpy
        directions = 270 - numpy.arctan2( b1 , a1 ) * self._toDeg

        for ii,direction in enumerate(directions):
            if direction < 0:
                direction = direction + 360

            if direction > 360:
                direction = direction - 360
            directions[ii] = direction
        return directions

    def mean_direction(self):
        return self._direction(self.a1m,self.b1m)

    def _spread(self,a1,b1):
        import numpy
        return numpy.sqrt(  2 - 2 * numpy.sqrt( a1**2 + b1**2 ) ) * self._toDeg

    def mean_spread(self):
        return self._spread(self.a1m,self.b1m )

    def _get_peak_value(self, variable ):
        maxloc = numpy.argmax( self.Szz,1 )

        out = numpy.zeros( maxloc.shape )

        if len(variable.shape) == 2:
            for ii,index in enumerate( maxloc ):
                out[ii] = variable[ ii,index ]
        elif len(variable.shape) == 1:
            for ii, index in enumerate(maxloc):
                out[ii] = variable[index]

        return out

    def peak_direction(self):
        a1 = self._get_peak_value( self.a1 )
        b1 = self._get_peak_value( self.b1 )
        return self._direction( a1,b1)

    def peak_spreading(self):
        a1 = self._get_peak_value( self.a1 )
        b1 = self._get_peak_value( self.b1 )
        return self._spread( a1,b1 )

    def peak_frequency(self):
        return self._get_peak_value( self.f )

    def peak_period(self):
        return 1. / self.peak_frequency()

    def mean_period(self):
        return 1. / self._weighted_moment( self.f )

    def significant_wave_height(self):
        import numpy
        return 4.*numpy.sqrt(self._moment( 1.))

    def generate_text_file(self):
        import os
        hm0   = self.significant_wave_height()
        tm01  = self.mean_period()
        tp    = self.peak_period()
        dir   = self.mean_direction()
        pdir  = self.peak_direction()
        dspr  = self.mean_spread()
        pdspr = self.peak_spreading()

        with open(os.path.join( self.outpath,'bulkparameters.csv'),'w') as file:

            header = "# year , month , day, hour ,min, sec, milisec , Significant Wave Height, Mean Period, Peak Period, Mean Direction, Peak Direction, Mean Spreading, Peak Spreading\n"
            file.write(header)
            format = '%d, ' * 7 + '%6.2f, ' * 6 + '%6.2f \n'
            for ii in range( 0 , len(hm0)):
                #
                string = format % ( self.time[ii,0], self.time[ii,1],  self.time[ii,2],
                                    self.time[ii,3],self.time[ii,4],self.time[ii,5],
                                    self.time[ii,6],hm0[ii],tm01[ii],tp[ii],
                                    dir[ii],pdir[ii],dspr[ii],pdspr[ii]  )
                file.write( string )
                #
            #
        #

def main( path = None , outpath=None, outputFileType='CSV',
          spectra='all',suffixes=None,parsing=None,lfFilter=False,bulkParameters=True):
    """ 
    Combine selected SPOTTER output files  into CSV files. This 
    routine is called by __main__ and that calls in succession the separate 
    routines to concatenate, and parse files. 
    """
    import os


    #Check the version of Files
    versions =  getVersions( path )
    
    #The filetypes to concatenate
    if suffixes is None:
        #
        suffixes    = ['FLT','SPC','SYS','LOC','GPS','SST']
        #
    
        
    if parsing is None:
        #
        parsing=['FLT','SPC','LOC','SST']
        #
    #
        
    outFiles    = {'FLT':'displacement','SPC':'spectra','SYS':'system',
                       'LOC':'location','GPS':'gps','SST':'sst'}

    if path is None:
        #
        # If no path given, assume current directory
        #
        path = os.getcwd()
        #
    else:
        #
        path = os.path.abspath(path)
        #    
    
    if (outpath is None):
        #
        outpath = path
        #
    else:
        outpath =  os.path.abspath(outpath)
        #
    #

    #Which spectra to process
    if spectra=='all':
        #
        outputSpectra = ['Szz','a1','b1','a2','b2','Sxx','Syy','Qxz','Qyz','Cxy']
        #
    else:
        #
        outputSpectra = [spectra]
        #
    #
    #
    # Loop over versions
    outp = outpath
    #
    for index,version in enumerate(versions):
        #
        if len(versions) > 1:
            #
            # When there are multiple conflicting version, we push output
            # to different subdirectories
            #
            outpath = os.path.join( outp,str(index) )
            #
        else:
            #            
            outpath = outp
            #
        #
            
        if not os.path.exists(outpath):
            #
            os.makedirs(outpath)
            #
        #            
        
        for suffix  in suffixes:
            #
            fileName = os.path.join( outpath , outFiles[suffix] + '.csv' )
            # 
            # For each filetypem concatenate files to intermediate CSV files...

            print( 'Concatenating all ' + suffix + ' files:')
            if not (cat(path=path, outputFileType='CSV',Suffix=suffix,
                    outputFileName=fileName,
                    versionFileList=version['fileNumbers'])
                    ):
                #
                continue
                #
            #
            
            #
            # ... once concatenated, process files further (if appropriate)
            #
            if suffix in parsing:
                #
                if suffix in ['FLT','LOC','GPS','SST']:
                    #
                    #parse the mean location/displacement files; 
                    #this step transforms unix epoch to date string.
                    #
                    parseLocationFiles(inputFileName = fileName,kind=suffix,
                            outputFileName = fileName,
                            outputFileType=outputFileType,
                            versionNumber=version['number'],
                            IIRWeightType=version['IIRWeightType'])
                    #
                elif suffix in ['SPC']:
                    #
                    #parse the mean location/displacement files; this step 
                    #extract relevant spectra (Szz, Sxx etc.) from the bulk 
                    #spectral file
                    parseSpectralFiles(inputFileName = fileName,
                                outputPath=outpath,
                                outputFileType=outputFileType,
                                outputSpectra=outputSpectra,lfFilter=lfFilter,
                                versionNumber=version['number'])
                    os.remove( fileName )
                    #
                #
            #parsing
            #
        #suffix
        #
        # Generate bulk parameter file
        if bulkParameters:
            spectrum = Spectrum(path=outpath,outpath=outpath)
            spectrum.generate_text_file()
    #Versions
    #

#end def



def log_errors( error ):
    global First
    if First:
        with open( 'error.txt','w') as file:
            file.write(error + '\n')
        First = False
    else:
        with open( 'error.txt','a') as file:
            file.write(error + '\n')
        First = False

def parseLocationFiles( inputFileName=None, outputFileName='displacement.CSV',
         kind='FLT', reportProgress=True, outputFileType='CSV',
        versionNumber=defaultVersion,IIRWeightType=defaultIIRWeightType ):
    """
    This functions loads all the gps-location data (located at *path*) from
    a spotter into one datastructure and saves the result as a CSV file 
    (*outputFileName*).
    """

    import os
    import pandas as pd
    import numpy as np
    import time
    #

    fname,ext = os.path.splitext(outputFileName)
    outputFileName = fname + '.' + extensions(outputFileType)
    

    #
    # Load location data into a pandas dataframe object
    if reportProgress:
        #
        print('Processing spotter displacement output')
        #
    #        

    #
    header = 'year,month,day,hour,min,sec,msec'
    if kind=='FLT':
        #
        # Read the data using pandas, and convert to numpy
        #
        data = pd.read_csv( inputFileName ,
                index_col=False , usecols=(1,2,3,4),error_bad_lines=False)
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values

        msk = np.isnan( data[:,0] )
        data = data[~msk,:]
        datetime    = epochToDateArray(data[:,0])
        data        = data[:,1:4]/1000.
        #
        # Apply phase correction to displacement data
        #
        if applyPhaseCorrection and versionNumber>=applyPhaseCorrectionFromVersionNumber:
            #
            print('- IIR phase correction using weight type: ', str(IIRWeightType ) )
            for ii in range(0,3):
                #
                data[:,ii] = applyfilter( data[:,ii] , 'backward' , versionNumber, IIRWeightType )
                #
            #        
        data        = np.concatenate( (datetime,data) , axis=1 )
            
        fmt = '%i,'*7  + '%.5e,%.5e,%.5e'
        header=header+', x (m), y(m), z(m)'
    elif kind=='SST':
        #
        # Read the data using pandas, and convert to numpy
        #
        data = pd.read_csv(inputFileName,
                           index_col=False, usecols=(0, 1))
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values

        msk = np.isnan(data[:, 0])
        data = data[~msk, :]
        datetime = epochToDateArray(data[:, 0])
        data = data[:, 1]
        data = np.concatenate((datetime, data[:,None]), axis=1)

        fmt = '%i,' * 7 + '%5.2f'
        header = header + ', T (deg. Celcius)'
    elif kind=='GPS':
        #
        # DEBUG MODE GPS
        #
        #
        data = pd.read_csv( inputFileName ,
                index_col=False , usecols=(1,2,3,4,5,6,7,8,9))
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values
        datetime    = epochToDateArray(data[:,0].tolist())

        data[:,1]  = data[:,1] + data[:,2] / 6000000.
        data[:,2]  = data[:,3] + data[:,4] / 6000000.
        data = np.concatenate( ( data[:,1:3] , data[:,5:] ), axis=1 ) 
        data        = np.concatenate( (datetime,data) , axis=1 )

        fmt = '%i,'*7 +'%13.8f' * 5 + '%13.8f'
        header=header+', latitude (decimal degrees),longitude (decimal ' + \
            'degrees),elevation (m),SOG (mm/s),COG (deg*1000),Vert Vel (mm/s)'
        #        
        #
    else:
        #
        data = pd.read_csv( inputFileName ,
                index_col=False , usecols=(0,1,2,3,4))
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values
        msk = np.isnan(data[:, 0])
        data = data[~msk, :]
        datetime    = epochToDateArray(data[:,0].tolist())

        data[:,1]  = data[:,1] + data[:,2] / 6000000.
        data[:,2]  = data[:,3] + data[:,4] / 6000000.
        data = data[:,1:3]
        data        = np.concatenate( (datetime,data) , axis=1 )

        fmt = '%i,'*7 +'%13.8f,%13.8f'
        header=header+', latitude (decimal degrees),longitude (decimal degrees)'
        #
    #

    


    if outputFileType.lower() in ['csv','gz']:
        #
        np.savetxt(outputFileName ,
            data ,fmt=fmt,
            header=header)
        #
        #
    elif outputFileType.lower()=='matlab':
        #
        # To save to matlab .mat format we need scipy
        #
        try:
            import scipy
            from scipy import io                
            #
            if kind=='FLT':
                scipy.io.savemat( outputFileName ,
                    {'x':data[:,7].astype(np.float32),
                     'y':data[:,8].astype(np.float32),
                     'z':data[:,9].astype(np.float32),
                  'time':data[:,0:7].astype(np.int16)} )
            elif kind=='GPS':
                scipy.io.savemat( outputFileName ,
                    {'Lat':data[:,7].astype(np.float32),
                     'Lon':data[:,8].astype(np.float32),
                     'elevation':data[:,9].astype(np.float32),
                     'sog':data[:,10].astype(np.float32),
                     'cog':data[:,11].astype(np.float32),
                     'w':data[:,12].astype(np.float32),                     
                     'time':data[:,0:7].astype(np.int16)} )                
            else:
                scipy.io.savemat( outputFileName ,
                    {'Lat':data[:,7].astype(np.float32),
                     'Lon':data[:,8].astype(np.float32),
                  'time':data[:,0:7].astype(np.int16)} )                
            #
        except ImportError:
            #
            raise Exception('To save as a matfile Scipy needs to be installed')
            #
        #
    elif outputFileType.lower()=='numpy':
        #
        if kind=='FLT':
            #
            np.savez( outputFileName ,
                x=data[:,7].astype(np.float32),
                y=data[:,8].astype(np.float32),
                z=data[:,9].astype(np.float32),
                time=data[:,0:7].astype(np.int16) )
            #
        else:
            #
            np.savez( outputFileName ,
                lat=data[:,7].astype(np.float32),
                lon=data[:,8].astype(np.float32),
                time=data[:,0:7].astype(np.int16) )
            #
        #
    #endif
    #
    return( None )
    #
#end def

def epochToDateArray( epochtime ):
    #
    import numpy as np
    import time
    
    datetime   = np.array( [ list(time.gmtime(x))[0:6] for x in epochtime ])
    milis      = np.array( [ ( 1000 * (  x-np.floor(x) ) ) for x in epochtime ])
    return(np.concatenate( (datetime,milis[:,None]),axis=1))
#
    

def parseSpectralFiles(   inputFileName=None, outputPath = None,
                          outputFileNameDict = None,
                          spectralDataSuffix='SPC', reportProgress=True      ,
                          nf=128                  , df=0.009765625           ,
                          outputSpectra=None      , outputFileType='CSV'     ,
                          lfFilter=False,versionNumber=defaultVersion):

    #
    # This functions loads all the Spectral data (located at *path*) from
    # a spotter into one datastructure and saves the result as a CSV file
    # (*outputFileName*).
    #
    import os
    import pandas as pd
    import numpy as np
    import time
    
    def checkKeyNames(key,errorLocation):
        # Nested function to make sure input is insensitive to capitals,
        # irrelevant permutations (Cxz vs Czx), etc
        if key.lower() == 'szz':
            out = 'Szz'
        elif key.lower() == 'syy':
            out = 'Syy'
        elif key.lower() == 'sxx':
            out = 'Sxx'                
        elif key.lower() in ['cxz','czx']:
            out = 'Cxz'
        elif key.lower() in ['qxz','qzx']:
            out = 'Qxz'
        elif key.lower() in ['cyz','czy']:
            out = 'Cyz'
        elif key.lower() in ['qyz','qzy']:
            out = 'Qyz'
        elif key.lower() in ['cxy','cyx']:
            out = 'Cxy'                
        elif key.lower() in ['qxy','qyx']:
            out = 'Qxy'
        elif key.lower() in ['a1','b1','a2','b2']:
            out = key.lower()
        else:
            raise Exception('unknown key: ' + key + ' in ' + errorLocation)
        return(out)
    #end def

    outputFileName= {'Szz':'Szz.CSV','Cxz':'Cxz.CSV','Qxz':'Qxz.CSV',
                     'Cyz':'Cyz.CSV','Qyz':'Qyz.CSV','Cxy':'Cxy.CSV',
                     'Qxy':'Qxy.CSV','Syy':'Syy.CSV','Sxx':'Sxx.CSV',
                     'a1':'a1.CSV','b2':'b2.CSV','b1':'b1.CSV','a2':'a2.CSV'}

    
    # Rename output file names for given variables if a dict is given
    if outputFileNameDict is not None:
        #
        for key in outputFileNameDict:
            #
            keyName = checkKeyNames(key,'output file names')
            outputFileName[ keyName ] = outputFileNameDict[ key ]
            #
        #
    #

    for key in outputFileName:
        #
        fname,ext = os.path.splitext(outputFileName[key])
        outputFileName[key] = fname + '.' + extensions(outputFileType)
        #
    #
        
    # The output  files given by the script; per defauly only Szz is given, but
    # can be altered by user request
    if outputSpectra is None:
        #
        outputSpectra = ['Szz']
        #
    else:
        #
        # For user requested output, make sure variables are known and have
        #correct case/permutations/etc.
        for index,key in enumerate(outputSpectra):
            #
            keyName = checkKeyNames(key,'output list')
            outputSpectra[index] = keyName
        #
    #
    

    #
    # Load spectral data into a pandas dataframe object
    if reportProgress:
        #
        print('Processing spotter spectral output')
        #
    #
    if versionNumber in [0,2]:
        startColumnNumber = {'Szz':5  ,'Cxz':7,
                         'Qxz':13,'Cyz':8,
                         'Qyz':14,'Sxx':3,
                         'Syy':4,  'Cxy':6,
                         'Qxy':12}
        stride = 12
    else:
        #
        # Column ordering changed from v1.4.1 onwards; extra columns due
        # to cross-correlation filter between z and w
        #
        # Column order is now:
        #
        #   type,millis,[0] t0_GPS_Epoch_Time(s), [1] tN_GPS_Epoch_Time(s), 
        #   [2] ens_count, [3] Sxx_re, [4] Syy_re, [5] Szz_re,[6] Snn_re,
        #   [7] Sxy_re,[8] Szx_re,[9] Szy_re,[10] Szn_re,[11] Sxx_im,
        #   [12] Syy_im,[13] Szz_im,[14] Snn_im,[15] Sxy_im,[16] Szx_im,
        #   [17] Szy_im,[18] Szn_im
        #
        # Note that since first two columns are ignorded counting starts from 0
        # at t0_GPS_Epoch_Time(s)
        #
        startColumnNumber = {'Szz':5  ,'Cxz':8,
                            'Qxz':16,'Cyz':9,
                            'Qyz':17,'Sxx':3,
                            'Syy':4,  'Cxy':7,
                            'Qxy':15, 'Snn':6,
                            'Czn':10, 'Qzn':18}
        stride = 16
        #
    #
    # Read csv file using Pandas - this is the only section in the code
    # still reliant on Pandas, and only there due to supposed performance
    # benifits.
    tmp = pd.read_csv( inputFileName ,
                index_col=False , skiprows=[0],  header=None,
                    usecols=tuple(range(2,5+stride*nf)) )
    
    # Ensure the dataframe is numeric, coerce any occurences of bad data
    # (strings etc) to NaN and return a numpy numerica array
    tmp = tmp.apply(pd.to_numeric, errors='coerce').values
      
    datetime    = epochToDateArray(tmp[:,0])
    ensembleNum = tmp[:,2] * 2
    data = {}
    for key in startColumnNumber:
        #
        # Convert to variance density and change units to m2/hz (instead
        # of mm^2/hz)
        #
        data[key] = tmp[: , startColumnNumber[key]::stride ]/( 1000000. * df )
        #
        # Set low frequency columns to NaN
        #
        data[key][:,0:3] = np.nan
        #
            
    # Calculate directional moments from data (if requested). Because these are
    # derived quantities these need to be included in the dataframe a-postiori
    if any( [ x in ['a1','b1','a2','b2'] for x in outputSpectra ] ):
        #
        with np.errstate(invalid='ignore',divide='ignore'):
            #Supress divide by 0; silently produce NaN
            data['a1'] = data['Qxz'] / np.sqrt( ( data['Szz'] * (
                data['Sxx'] + data['Syy'] ) ) )
            data['a2'] = ( data['Sxx'] - data['Syy'] ) / (
                data['Sxx'] + data['Syy'] )
            data['b1'] = data['Qyz'] / np.sqrt( ( data['Szz'] * (
                data['Sxx'] + data['Syy'] ) ) )
            data['b2'] = 2. * data['Cxy'] /  (
                data['Sxx'] + data['Syy'] )                      
        #

        for key in ['a1','b1','a2','b2']:
            #
            # If energies are zeros, numpy produces infinities
            # set to NaN as these are meaningless
            #
            data[key][ np.isinf(data[key] ) ]= np.nan
            data[key][ np.isnan(data[key] ) ]= np.nan
    #

    if lfFilter:
        #
        # Filter lf-noise
        data = lowFrequencyFilter( data )
        #
    #
        
    for key in data:
        #
        data[key] = np.concatenate( (datetime,
                                         ensembleNum[:,None],data[key]),axis=1 )
        #
    #

    # construct header for use in CSV
    header = 'year,month,day,hour,min,sec,milisec,dof'
    freq = np.array(list( range( 0 , nf ) )) * df
    #
    for f in freq:
        #        
        header = header +','+ str(f)
        #    
    #
        
    #
    # write data to requested output format
    #
    for key in outputSpectra:
        #
        fmt = '%i , ' * 8 + ('%.3e , ' * (nf-1)) + '%.3e'

        if outputFileType.lower()=='csv':
            #
            if outputFileType.lower() in ['csv','gz']:
                #
                np.savetxt(os.path.join( outputPath , outputFileName[key]) ,
                           data[key], fmt=fmt,
                           header=header)
                #            
            #
        elif outputFileType.lower()=='matlab':
            #
            # To save to matlab .mat format we need scipy
            #
            try:
                import scipy
                from scipy import io                
                #
                mat = data[key]
                scipy.io.savemat(
                        os.path.join( outputPath , outputFileName[key]),
                        {'spec':mat[:,8:].astype(np.float32),
                        'time':mat[:,0:7].astype(np.int16),
                        'frequencies':freq.astype(np.float32),
                        'dof':mat[:,7].astype(np.int16) } )
                #
            except ImportError:
                #
                raise Exception('Saving as a matfile requires Scipy')
                #
            #
        elif outputFileType.lower()=='numpy':
            #
            mat = data[key]         
            np.savez(os.path.join( outputPath , outputFileName[key]),
                        spec=mat[:,8:].astype(np.float32),
                        time=mat[:,0:7].astype(np.int16),
                        frequencies=freq.astype(np.float32),
                        dof=mat[:,7].astype(np.int16) )
            #
        #
    #
    #
    return( None )
    #
#end def

def lowFrequencyFilter( data ):
    '''
    function to perform the low-frequency filter
    '''   
    import numpy as np
    #
    with np.errstate(invalid='ignore',divide='ignore'):
        # Ignore division by 0 etc (this is caught below)
        Gxz = (data['Cxz']**2 + data['Qxz']**2) / ( data['Sxx'] * data['Szz'] )
        Gyz = (data['Cyz']**2 + data['Qyz']**2) / ( data['Syy'] * data['Szz'] )
        #
    #
    phi = 1.5 * np.pi - np.arctan2( data['b1'] , data['a1'] )
    G   = ( np.sin( phi) **2 * Gxz + np.cos( phi) **2 * Gyz )
    G[ np.isnan(G)] = 0.

    I  = np.argmax( G           , axis=1 )

    names =  ['Szz','Cxz','Qxz','Cyz','Qyz','Sxx','Syy','Cxy','Qxy']
    for key in names:
        #
        for jj in range( 0 , G.shape[0] ):
            #
            data[ key ][ jj , 0 : I[jj] ] = \
              data[ key ][ jj , 0 : I[jj] ] * G[jj,0 : I[jj]]           
            #
        #
    #
    return( data )
    #
#end def

def getFileNames( path , suffix , message,versionFileList=None ):
    #
    # This function returns all the filenames in a given *path*
    # that conform to ????_YYY.CSV where YYY is given by *suffix*.
    #
    import os
    import fnmatch
    
    if path is None:
        #
        # If no path given, assume current directory
        #
        path = os.getcwd()
        #
    else:
        #
        path = os.path.abspath(path)
        #
    #

        
    synonyms = [suffix]
    if suffix == 'LOC':
        #
        synonyms.append('GPS')
        #
    #
    # Get the file list from the directory, and select only those files that
    # match the spotter output filename signature
    #

    if versionFileList is not None:
        #
        # Only add filenames that are in the present version file number list
        #
        fileNames = []
        for filename in sorted(os.listdir(path)):
            #
            if fnmatch.fnmatch(filename, '????_' + suffix + '.CSV'):
                #
                num , ___ = filename.split('_')
                num = num.strip()
                if num in versionFileList:
                    #
                    fileNames.append(filename)
                    #
                #
            #
        #
    else:
        #
        fileNames =  [ filename for filename in sorted(os.listdir(path))
                if fnmatch.fnmatch(filename, '????_' + suffix + '.CSV') ]    
    #
    # Are there valid spotter files?
    #    
    if len( fileNames ) < 1:
        #
        # No files found; raise exception and exit
        #
        print('  No ' + message + ' data files available.')
        #
    #
    return( path , fileNames )
    #
#

def extensions( outputFileType ):
    #
    ext = {'csv':'csv','matlab':'mat','numpy':'npz','pickle':'pickle','gz':'gz'}
    if outputFileType.lower() in ext:
        #
        return( ext[outputFileType.lower()] )
        #
    else:
        #
        raise Exception('Unknown outputFileType; options are:'
                            + 'numpy , matlab , pickle , csv')
        #
    #
#
  
def cat( path = None, outputFileName = 'displacement.CSV', Suffix='FLT',
             reportProgress=True, outputFileType='CSV',versionFileList=None ):
    """
    This functions concatenates raw csv files with a header. Only for the first 
    file it retains the header. Note that for SPEC files and SST files special
    processing is done. Specifically, for SST files we map the millis timebase
    onto the epochtime base using a relation estimated from the FLT files.
    """
    import os
    import pandas as pd
    import numpy as np

    def get_epoch_to_milis_relation( sst_file ):
        #
        # This function gets the relation between milis and epoch from the
        # FLT file. This assumes FLT exist, otherwise we get an error
        #

        head,tail = os.path.split( sst_file )
        tail = tail.replace('SST','FLT')

        flt_file = os.path.join(head,tail)
        data = pd.read_csv( flt_file ,index_col=False , usecols=(0,1))
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values
        msk = np.isnan( data[:,0] )
        for ii in range( 0, data.ndim ):
            msk = np.isfinite(data[:,ii])
            data = data[msk,:]
        data = data[msk,:]
        millis = data[:,0]
        epochs = data[:,1]

        ii  = numpy.argmax( millis)

        if ii < 10:
            raise Exception('Roll-over in millis')

        def milis_to_epoch(milis_in):
            return int(epochs[0] + ( milis_in - millis[0] ) \
                   * ( epochs[ii] - epochs[0] ) / ( millis[ii] - millis[0] ))

        return milis_to_epoch

    def process_sst_lines( lines, infile ):
        #
        # Get the function that maps milis to epochs
        #
        # max int used for roll-over; spotter millis clock resets after reaching
        # the max ints.
        max = 4294967295


        #Get the millis to epoch mapping from the FLT file
        milis_to_epoch = get_epoch_to_milis_relation(infile)

        # Do a line by line processing, replacing millis with epochtime from
        # the mapping
        lines = lines.split('\n')
        outlines = []                #Store the output lines
        previousvalue = 0
        for line in lines:
            if 'millis' in line:
                outlines.append((line))
            else:
                #
                data  = line.split(',')
                # last line can be empty, check if there are two entries (as expected)
                if len(data) == 2:

                    # Look at the delta, millis should be monotonically increasing
                    # unless we hit roll-over
                    delta = int(data[0]) - previousvalue

                    # If the delta is negative, wrap the value
                    if delta < -4294000000:
                        delta = delta + max

                    # New value from (potentially) corrected delta
                    value = previousvalue + delta

                    #Convert to epochtime from mapping
                    epoch = milis_to_epoch( value )

                    outlines.append(str( epoch ) + ' , ' + data[1])
                    previousvalue = value



        return '\n'.join(outlines) + '\n'

    import os
    import gzip
    #

    def modeDetection( path , filename ):
        #
        # Here we detect if we are in debug or in production mode, we do this
        # based on the first few lines; in debug these will contain either
        # FFT or SPEC, whereas in production only SPECA is encountered
        import os
        
        mode = 'production'
        with open(os.path.join( path,filename) ) as infile:
            #
            jline = 0 
            for line in infile:
                #
                if ( line[0:3]=='FFT' or line[0:5]=='SPEC,'):
                    mode = 'debug'
                    break
                jline = jline+1
                
                if jline>10:
                    break
        return(mode)
    
    def find_nth(haystack, needle, n):
        #
        # Function to find nth and nth-1 occurences of a needle in the haystack
        #
        start = haystack.find(needle)
        prev  = start
        while start >= 0 and n > 1:
            #
            prev  = start
            start = haystack.find(needle, start+len(needle))
            n -= 1
            #
        return (start,prev)
        #
    #end nested function
    #        
    #
    # Get a list of location filenames and the absolute path 
    path , fileNames = getFileNames( path=path , suffix=Suffix,
                        message='_'+Suffix,versionFileList=versionFileList )
    #
    if len(fileNames) == 0:
        return(False)
    
    fname,ext = os.path.splitext(outputFileName)
    outputFileName = fname + '.' + extensions(outputFileType)
    #
    if outputFileType.lower()=='gz':
        #
        compress=True
        #
    else:
        #
        compress=False
        #    
    #
    if compress:
        #
        outfile = gzip.open(outputFileName,'wb')
        #
    else:
        #
        outfile = open(outputFileName,'w')
        #
    #
    #    
    #
    for index,filename in enumerate(fileNames):
        #
        if reportProgress:
            #
            print( '- ' + filename + ' (File {} out of {})'.format(
                index+1, len(fileNames) ) )
            #
        #
        
        if Suffix == 'SPC':
            #
            ip = 0
            prevLine = ''
            mode = modeDetection( path, filename)
            #

            with open(os.path.join( path,filename) ) as infile:
                try:        #
                    #lines = infile.readlines()
                    ii = 0
                    line = infile.readline()
                    #
                    if index==0:
                        #
                        outfile.write(line)

                    for line in infile:
                        #
                        # Read the ensemble counter
                        if (line[0:8]=='SPEC_AVG' or line[0:5]=='SPECA'
                                 or line[0:8]=='SPECA_CC'):
                            a,b=find_nth(line, ',', 5)
                            ii = int(line[b+1 : a])
                            #

                            if mode=='production':
                                #
                                outfile.write(line)
                                ip       = ii
                                #
                            else:
                                #
                                #
                                # Debug spectral file, contains all averages,
                                # only need last.
                                #
                                if line[0:8]=='SPECA_CC':
                                    #
                                    outfile.write(line)
                                    ip = 0
                                    #
                                elif ii < ip:
                                    #
                                    outfile.write(prevLine)
                                    ip = 0
                                    #
                                else:
                                    #
                                    ip       = ii
                                    prevLine = line
                                    #
                                #end if
                                #
                            #end if
                            #
                        #end if
                        #
                    #end for line
                    #
                except:
                    message = "- ERROR:, file " + os.path.join( path,filename) + " is corrupt"
                    log_errors(message)
                    print(message )
            #end with
            #
        else:
            #
            with open(os.path.join( path,filename) ) as infile:
                #
                try:
                    lines = infile.read()
                    #
                    if index==0:
                        #
                        ind = lines.find('\n') + 1
                        #
                    else:
                        #
                        lines = lines[ind:]
                        #
                    #
                    # If SST file, map millis onto epochs
                    if Suffix == 'SST':
                        lines = process_sst_lines(lines, filename)

                    if compress:
                        #
                        lines=lines.encode('utf-8')
                        #
                    else:
                        # Strip dos newline char
                        lines = lines.replace('\r','')


                    outfile.write(lines)
                except:
                    message = "- ERROR:, file " + os.path.join( path,filename) + " is corrupt"
                    log_errors(message)
                    print(message )
            #
        #
    #
    outfile.close()
    return( True )
    #
#end def

    
def validCommandLineArgument( arg ):
    #
    import sys
    out = arg.split('=')

    if not (len(out) == 2):
        #
        print('ERROR: Unknown commandline argument: ' + arg)
        sys.exit()
        #
    key,val = out
    
    if key.lower() in ['path']:
        #
        key = 'path'
        #
    elif  key.lower() in ['outputfiletype']:
        #
        key = 'outputFileType'
        #
    elif key.lower() in ['spectra']:
        #
        key = 'spectra'
        #
    elif key.lower() in ['lffilter']:
        #
        key = 'lfFilter'
        #
    elif key.lower() in ['bulkparameters']:
        #
        key = 'bulkParameters'
        #
    else:
        #
        print('ERROR: unknown commandline argument ' + key)
        sys.exit()
        #
    return( key,val)
    #

def getVersions( path ):
    """
     This function retrieves sha from sys filenames; if no sha is present
     within the first 20 lines, it is assumed the previous found sha is
     active. The output is a version list; each entry in the version list
     is a dict that contains all file prefixes (0009 etc.) that can be
     processed in the same way (this may go across firmware versions).
    """
    
    import os

    # Get sys files
    path,fileNames = getFileNames( path , 'SYS' , 'system' )
    #
    def latestVersion():
        #
        ordinal = - 1
        for key in ordinalVersionNumber:
            #
            if ordinalVersionNumber[key][1] > ordinal:
                #
                latVer = key
                #
            #
        #
        return(latVer)
        #
    #end def
    #
    first = True
    version = []
    #
    # Loop over all the _SYS files
    for index,filename in enumerate(fileNames):
        #        
        foundSha = False
        foundIIRWeightType = False
        IIRWeightType = defaultIIRWeightType
        #
        #Check if there is a sha in first 80 lines
        with open(os.path.join( path,filename) ) as infile:
            #
            jline = 0
            for line in infile:
                if 'SHA' in line:
                    ___ , sha = line.split(':')
                    sha = sha.strip()
                    foundSha = True
                elif 'iir weight type' in line:
                    ___ , IIRWeightType = line.split(':')
                    IIRWeightType = int(IIRWeightType.strip())
                    foundIIRWeightType = True
                    #
                jline += 1
                if (foundSha and foundIIRWeightType) or jline>80:
                    #
                    break
                #
        #
        # If we found a SHA, check if it is valid
        if foundSha:
            #
            # Is it a valid sha? 
            #                
            if not sha in ordinalVersionNumber:
                #
                # If not - parse using the latest version
                sha = latestVersion()
            #

        #
        #Valid sha, so what to do?
        if foundSha and first:
            #
            # this the first file, and we found a sha
            #
            version.append( {'sha':[sha],
                             'version':[supportedVersions[sha]],
                             'ordinal':[ordinalVersionNumber[sha]],
                             'number':ordinalVersionNumber[sha][1],
                             'IIRWeightType':IIRWeightType,
                             'fileNumbers':[] })
            first = False            
            #
        elif not foundSha and first:
            #
            # this is the first file, but no sha - we will try to continue
            # under the assumption that the version corresponds to the
            # latest version - may lead to problems in older version
            print('WARNING: Cannot determine version number')
            sha = latestVersion()
            version.append( {'sha':[sha],
                             'version':[supportedVersions[sha]],
                             'ordinal':[ordinalVersionNumber[sha]],
                             'number':ordinalVersionNumber[sha][1],
                             'IIRWeightType':IIRWeightType,
                             'fileNumbers':[] })
            first = False
            #
        elif foundSha and not first:
            #
            # We found a new sha, check if it is the same as previous found sha
            # and if so just continue
            #
            if not ( sha in version[-1]['sha'] and version[-1]['IIRWeightType']==IIRWeightType) :
                #
                # if not, check if this version is compatible with the previous
                # found version
                if ordinalVersionNumber[sha][1] == version[-1]['number'] and version[-1]['IIRWeightType']==IIRWeightType:
                    #
                    # If so, append the sha/version
                    #
                    version[-1]['sha'].append(sha)
                    version[-1]['ordinal'].append(ordinalVersionNumber[sha])
                    version[-1]['version'].append(supportedVersions[sha])
                    #
                else:
                    #
                    # Not Compatible, we add a new version to the version list
                    # that has to be processed seperately
                    version.append( {'sha':[sha],
                                     'version':[supportedVersions[sha]],
                                     'ordinal':[ordinalVersionNumber[sha]],
                                     'number':ordinalVersionNumber[sha][1],
                                     'IIRWeightType':IIRWeightType,
                                     'fileNumbers':[] })
                    #                             
                #
            #
        #
        entry , ___ = filename.split('_')
        #
        # Add file log identifier (e.g. 0009_????.csv, with entry = '0009')
        #
        version[-1]['fileNumbers'].append( entry )                        
        #
    #end for filenames
    #
    return version
#
def filterSOS(versionNumber,IIRWeightType):
    #
    import numpy as np
    #second order-sections coeficients of the filter

    if versionNumber < 1:
        #
        sos =0.
        return sos
        #
    elif versionNumber in [1,2]:
        #
        if IIRWeightType == 0:
            #Type A
            lp = {'a1': -1.8514229621,
                  'a2':  0.8578089736,
                  'b0': 0.8972684452 ,
                  'b1': -1.7945369122 ,
                  'b2': 0.8972684291  }             
            hp = {'a1':-1.9318795385 ,
                  'a2':0.9385430645 ,
                  'b0':1.0000000000 ,
                  'b1':-1.9999999768 ,
                  'b2':1.0000000180 }          
        elif IIRWeightType == 1:
            #Type B           
            lp = {'a1': 1.9999999964  ,
                  'a2': 0.9999999964  ,
                  'b0': 0.9430391609  ,
                  'b1': -1.8860783217 ,
                  'b2': 0.9430391609  }              
            hp = {'a1':-1.8828311523,
                  'a2':0.8893254984 ,
                  'b0':1.0000000000 ,
                  'b1':2.0000000000 ,
                  'b2':1.0000000000 }            
        elif IIRWeightType == 2:
            #Type C         
            lp = {'a1': 1.1375322034,
                  'a2': 0.4141775928,
                  'b0': 0.6012434213 ,
                  'b1': -1.2024868427 ,
                  'b2': 0.6012434213  }
            hp = {'a1':-1.8827396569 ,
                  'a2':0.8894088696  ,
                  'b0':1.0000000000 ,
                  'b1':2.0000000000 ,
                  'b2':1.0000000000 }            
        #
    #
    sos = [ [ lp['b0'],lp['b1'],lp['b2'],1.000000,lp['a1'],lp['a2'] ],
            [ hp['b0'],hp['b1'],hp['b2'],1.000000,hp['a1'],hp['a2'] ] ]
    sos = np.array( sos )
    return sos
            
def applyfilter( data , kind , versionNumber, IIRWeightType ):
    import numpy as np
    from scipy import signal
    #
    # Apply forward/backward/filtfilt sos filter
    #
    # Get SOS coefficients
    sos = filterSOS( versionNumber,IIRWeightType)
    #
    if kind=='backward':
        #
        directions = ['backward']
        #, axis=0
    elif kind=='forward':
        #
        directions = ['forward']
        #
    elif kind=='filtfilt':
        #
        directions = ['forward','backward']

    res = data
    for direction in directions:
        #
        if direction=='backward':
            #
            res = np.flip( res, axis=0 )
            #
    
        res = signal.sosfilt(sos, res,axis=0)

        if direction=='backward':
            #
            res = np.flip( res, axis=0 )
            #

    return res

First = True
if __name__ == "__main__":
    #
    # execute only if run as a script
    #
    import sys

    narg      = len( sys.argv[1:] )
    
    if narg>0:
        #
        #parse and check command line arguments
        arguments = dict()
        for argument in sys.argv[1:]:
            #
            key,val = validCommandLineArgument( argument )
            arguments[key]=val            
            #
        #
    else:
        #
        arguments = dict()
        #
    #
    main(**arguments)



