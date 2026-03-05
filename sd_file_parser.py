import os
import sys
import argparse
import time
from glob import glob

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import savemat

def parse_spotter_files( input_path = None , output_path=None, output_format='CSV',
          spectra='all', lf_filter = False, include_n_channel=False):
    """ 
    Combine selected SPOTTER output files  into CSV files. This 
    routine is called by cli_main and that calls in succession the separate 
    routines to concatenate, and parse files. 
    """

    #Check the version of Files
    versions =  getVersions( input_path )

    #The filetypes to concatenate
    file_types = [
        'FLT',
        'SPC',
        'SYS',
        'LOC',
        'GPS',
        'SST',
    ]

    parsing = [
        'FLT',
        'SPC',
        'LOC',
        'SST',
    ]

    if any( [ version['number'] < 3 for version in versions ] ):
        print("WARNING: This parser version no longer supports legacy SST files (firmware 1.12.x and below).")
        print("         If you need to parse legacy SST files, please use the parser in the legacy/ subdirectory.")
        print("Press enter to confirm (or q to quit):")
        log_errors("Warned about skipping legacy SST parsing.")
        answer = input()
        if answer.lower() == 'q':
            sys.exit(1)
        for collection in file_types, parsing:
            collection.remove('SST')

    outFiles    = {'FLT':'displacement','SPC':'spectra','SYS':'system',
                       'LOC':'location','GPS':'gps','SST':'sst'}

    if input_path is None:
        #
        # If no path given, assume current directory
        #
        input_path = os.getcwd()
        #
    else:
        #
        input_path = os.path.abspath(input_path)
        #    
    
    if output_path is None:
        #
        output_path = os.path.join(input_path, 'processed')
        #
    else:
        output_path =  os.path.abspath(output_path)
        #
    #

    #Which spectra to process
    if spectra=='all':
        #
        outputSpectra = ['Szz','a1','b1','a2','b2','Sxx','Syy','Qxz','Qyz','Cxy']
        if include_n_channel:
            outputSpectra.extend(['Snn', 'Czn'])
        #
    else:
        #
        outputSpectra = [spectra]
        #
    #
    #
    # Loop over versions
    outp = output_path
    if len(versions) == 1:
        # if there is only a single version- we allow all files to be parsed.
        # this is a clutch to account for the fact that sys files are not
        # guaranteed to be written. In general assuming everything is the
        # same version seems safe- allowing to parse multiple different
        # versions is perhaps something we want to stop supporting as it adds
        # a lot of fragile logic.
        versions[0]['fileNumbers'] = None

    #
    for index,version in enumerate(versions):
        #
        if len(versions) > 1:
            #
            # When there are multiple conflicting version, we push output
            # to different subdirectories
            #
            output_path = os.path.join( outp,str(index) )
            #
        else:
            #            
            output_path = outp
            #
        #
            
        if not os.path.exists(output_path):
            #
            os.makedirs(output_path)
            #
        #            

        for suffix in file_types:
            #
            fileName = os.path.join( output_path , outFiles[suffix] + '.csv' )
            # 
            # For each filetype, concatenate files to intermediate CSV files...

            print( 'Concatenating all ' + suffix + ' files:')
            if not (cat(path=input_path, outputFileType='CSV',Suffix=suffix,
                    outputFileName=fileName,
                    versionFileList=version['fileNumbers'])):
                #
                continue
                #
            #
            
            #
            # ... once concatenated, process files further (if appropriate)
            #
            if suffix in parsing:
                #
                if suffix in [
                    'FLT',
                    'LOC',
                    'GPS',
                    'SST',
                ]:
                    #
                    #parse the mean location/displacement files; 
                    #this step transforms unix epoch to date string.
                    #
                    try:
                        parseLocationFiles(inputFileName = fileName, kind=suffix,
                            outputFileName = fileName,
                            outputFileType=output_format,
                            include_n_channel=include_n_channel,
                            versionNumber=version['number'],
                            IIRWeightType=version['IIRWeightType'])
                    except OSError as e:
                        print(f"Error in parseLocationFiles() while parsing {fileName}: {e}")
                        raise
                    #
                elif suffix in ['SPC']:
                    #
                    #parse the mean location/displacement files; this step 
                    #extract relevant spectra (Szz, Sxx etc.) from the bulk 
                    #spectral file
                    parseSpectralFiles(inputFileName = fileName,
                                outputPath=output_path,
                                outputFileType=output_format,
                                outputSpectra=outputSpectra,
                                lf_filter=lf_filter,
                                include_n_channel=include_n_channel,
                                versionNumber=version['number'])
                    os.remove( fileName )
                    #
                #
            #parsing
            #
        #suffix
        #
        # Generate bulk parameter file
        spectrum = Spectrum(path=output_path,outpath=output_path)
        spectrum.generate_text_file()
    #Versions
    #
    print("Done. Processed data saved in : " + output_path )
    return output_path

class Spectrum:
    _parser_files = {
        'Szz':  'Szz.csv',
        'a1':   'a1.csv',
        'b1':   'b1.csv',
        'Sxx':  'Sxx.csv',
        'Syy':  'Syy.csv',
        'Qxz':  'Qxz.csv',
        'Qyz':  'Qyz.csv'
    }
    _toDeg = 180. / np.pi

    def __init__(self,path,outpath):
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

            else:
                self._file_available[key] = False

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
        return np.mean( self._data['Qyz'],1 )

    def _Qxzm(self):
        return np.mean( self._data['Qxz'],1 )

    def _Sxxm(self):
        return np.mean( self._data['Sxx'],1 )

    def _Syym(self):
        return np.mean( self._data['Syy'],1 )

    def _Szzm(self):
        return np.mean( self._data['Szz'],1 )

    @property
    def f(self):
        return self._frequencies

    def _load_header(self):
        for key in self._parser_files:
            if self._file_available[key]:
                with open(os.path.join(self.path,self._parser_files[key]),'r') as file:
                    line = file.readline(  ).split(',')[8:]
                    self._frequencies = np.array([float(x) for x in line])
            break
        else:
            raise Exception('No spectral files available - please check the input directory')


    def _load_parser_output( self ):
        #
        for key in self._parser_files:
            if self._file_available[key]:
                data = np.loadtxt( os.path.join(self.path,self._parser_files[key]) , delimiter=',' )

                if data.ndim == 1:
                    # If there is only 1 line in the file, we need to add a dimension to the data as numpy returns a
                    # 1D array in this case.
                    data = data[None,:]

                self.time = data[ : , 0:8 ]
                self._data[key] = data[:,8:]
                mask = np.isnan( self._data[key] )
                self._data[key][mask] = 0.
                self._none = np.nan + np.zeros( self._data[key].shape )

        for key in self._parser_files:
            if not self._file_available[key]:
                self._data[key] = self._none


    def _moment(self , values ):
        df = np.mean(np.diff( self.f))
        E = self.Szz * values
        jstart =3
        return np.trapezoid(  E[:,jstart:],self.f[jstart:] ,1  )

    def _weighted_moment(self , values ):
        return self._moment( values ) / self._moment(1.)

    @property
    def a1m(self):
        if self._file_available['Sxx']:
            return self._Qxzm() / np.sqrt( self._Szzm() * ( self._Sxxm() + self._Syym() ))
        else:
            return self._weighted_moment( self.a1 )

    @property
    def b1m(self):
        if self._file_available['Sxx']:
            return self._Qyzm() / np.sqrt( self._Szzm() * ( self._Sxxm() + self._Syym() ))
        else:
            return self._weighted_moment( self.b1 )

    def _peak_index(self):
        return np.argmax( self.Szz,1 )

    def _direction(self,a1,b1):
        directions = 270 - np.arctan2( b1 , a1 ) * self._toDeg

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
        return np.sqrt(  2 - 2 * np.sqrt( a1**2 + b1**2 ) ) * self._toDeg

    def mean_spread(self):
        return self._spread(self.a1m,self.b1m )

    def _get_peak_value(self, variable ):
        maxloc = np.argmax( self.Szz,1 )

        out = np.zeros( maxloc.shape )

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
        return 4.*np.sqrt(self._moment( 1.))

    def generate_text_file(self):
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

#'SHA <-> version-number' relation
#(note that dual entry for 1.2.5/1.4.2 is due to update glitches)
supportedVersions    = {'1446ABC':'1.2.5','9BEADBE':'1.3.0','2FDC90':'1.4.1',
                        '928D2AE':'1.2.5','3D3CFF5':'1.?.?','A0F6980':'1.?.?',
                        '340b03f':'1.4.2','82755AE':'1.4.2','B218FBD':'1.5.1',
                        'E52AC4D':'1.5.2','1323D38':'1.5.3','73A3A4D0':'1.6.0',
                        '6171C497':'1.6.2','A98A2E52':'1.7.0','A4FAAEBA':'1.7.1',
                        'E7C7CD94':'1.8.0','412432D3':'1.9.0','9F438E3C':'1.9.1',
                        '97A11B27':'1.10.0','2992193B':'1.11.0','2569BD17':'1.11.1',
                        '81C1B398':'1.11.2','93CE0B95':'1.12.0','FE6412C3':'1.13.0'}

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
                        'FE6412C3':(19,3)
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
        versionNumber=defaultVersion,IIRWeightType=defaultIIRWeightType,
        include_n_channel=False ):
    """
    This functions loads all the gps-location data (located at *path*) from
    a Spotter into one datastructure and saves the result as a CSV file 
    (*outputFileName*).
    """
    #

    fname,ext = os.path.splitext(outputFileName)
    outputFileName = fname + '.' + extensions(outputFileType)

    #
    # Load location data into a pandas dataframe object
    if reportProgress:
        #
        print(f"Processing Spotter output - {kind}")
        #
    #        

    #
    header = 'year,month,day,hour,min,sec,msec'
    if kind=='FLT':
        #
        # Read the data using pandas, and convert to numpy
        #
        usecols = (1,2,3,4)
        if include_n_channel:
            try:
                data = pd.read_csv(
                    inputFileName,
                    index_col=False,
                    usecols=(1,2,3,4,5),
                )
            except ValueError:
                print("WARNING: include_n_channel requested, but outn(mm) was not found. Continuing without n channel.")
                data = pd.read_csv(
                    inputFileName,
                    index_col=False,
                    usecols=usecols,
                )
                include_n_channel = False
        else:
            data = pd.read_csv(
                inputFileName,
                index_col=False,
                usecols=usecols,
            )

        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.values

        msk = np.isnan( data[:,0] ) | np.isnan( data[:,1] ) | np.isnan( data[:,2] ) | np.isnan( data[:,3] )
        data = data[~msk,:]
        datetime    = epochToDateArray(data[:,0])
        if include_n_channel:
            data = data[:,1:5]/1000.
            n_dims = 4
        else:
            data = data[:,1:4]/1000.
            n_dims = 3
        #
        # Apply phase correction to displacement data
        #
        if applyPhaseCorrection and versionNumber>=applyPhaseCorrectionFromVersionNumber:
            #
            print('- IIR phase correction using weight type: ', str(IIRWeightType ) )
            for ii in range(0,n_dims):
                #
                data[:,ii] = applyfilter( data[:,ii] , 'backward' , versionNumber, IIRWeightType )
                #
            #        
        data        = np.concatenate( (datetime,data) , axis=1 )

        if include_n_channel:
            fmt = '%i,'*7  + '%.5e,%.5e,%.5e,%.5e'
            header=header+', x (m), y(m), z(m), n (m)'
        else:
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
        header = header + ', T (deg. Celsius)'
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
        data = data.values.astype(np.float64)
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
            #
            if kind=='FLT':
                n_filt_cols_no_n = 10
                matlab_data = {
                    'x':data[:,7].astype(np.float32),
                    'y':data[:,8].astype(np.float32),
                    'z':data[:,9].astype(np.float32),
                    'time':data[:,0:7].astype(np.int16),
                }
                if include_n_channel and data.shape[1] == n_filt_cols_no_n + 1:
                    matlab_data['n'] = data[:,10].astype(np.float32)
                savemat(outputFileName, matlab_data)
            elif kind=='GPS':
                savemat( outputFileName ,
                    {'Lat':data[:,7].astype(np.float32),
                     'Lon':data[:,8].astype(np.float32),
                     'elevation':data[:,9].astype(np.float32),
                     'sog':data[:,10].astype(np.float32),
                     'cog':data[:,11].astype(np.float32),
                     'w':data[:,12].astype(np.float32),                     
                     'time':data[:,0:7].astype(np.int16)} )                
            else:
                savemat( outputFileName ,
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
            np_data = {
                'x':data[:,7].astype(np.float32),
                'y':data[:,8].astype(np.float32),
                'z':data[:,9].astype(np.float32),
                'time':data[:,0:7].astype(np.int16),
            }
            if include_n_channel and data.shape[1] > 10:
                np_data['n'] = data[:,10].astype(np.float32)
            np.savez(outputFileName, **np_data)
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
    
    datetime   = np.array( [ list(time.gmtime(x))[0:6] for x in epochtime ])
    milis      = np.array( [ ( 1000 * (  x-np.floor(x) ) ) for x in epochtime ])
    return(np.concatenate( (datetime,milis[:,None]),axis=1))
#
    

def parseSpectralFiles( inputFileName=None, 
                        outputPath = None,
                        outputFileNameDict = None,
                        reportProgress=True,
                        nf=128, 
                        df=0.009765625,
                        outputSpectra=None, 
                        outputFileType='CSV',
                        lf_filter = False,
                        include_n_channel=False,
                        versionNumber=defaultVersion):

    #
    # This functions loads all the Spectral data (located at *path*) from
    # a Spotter into one datastructure and saves the result as a CSV file
    # (*outputFileName*).
    #

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
        elif key.lower() == 'snn':
            out = 'Snn'
        elif key.lower() in ['czn','cnz']:
            out = 'Czn'
        elif key.lower() in ['a1','b1','a2','b2']:
            out = key.lower()
        else:
            raise Exception('unknown key: ' + key + ' in ' + errorLocation)
        return(out)
    #end def

    outputFileName= {'Szz':'Szz.CSV','Cxz':'Cxz.CSV','Qxz':'Qxz.CSV',
                     'Cyz':'Cyz.CSV','Qyz':'Qyz.CSV','Cxy':'Cxy.CSV',
                     'Qxy':'Qxy.CSV','Syy':'Syy.CSV','Sxx':'Sxx.CSV',
                     'Snn':'Snn.CSV','Czn':'Czn.CSV',
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
        
    # The output files given by the script; per default only Szz is given, but
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
        print('Processing Spotter spectral output')
        #
    #
    legacyStartColumnNumber = {'Szz':5  ,'Cxz':7,
                         'Qxz':13,'Cyz':8,
                         'Qyz':14,'Sxx':3,
                         'Syy':4,  'Cxy':6,
                         'Qxy':12}
    # Column ordering changed from v1.4.1 onwards; extra columns due
    # to cross-correlation filter between z and w.
    extendedStartColumnNumber = {'Szz':5  ,'Cxz':8,
                            'Qxz':16,'Cyz':9,
                            'Qyz':17,'Sxx':3,
                            'Syy':4,  'Cxy':7,
                            'Qxy':15, 'Snn':6,
                            'Czn':10, 'Qzn':18}

    n_header_columns = 5 # type, millis, t0 epoch time, tN epoch time, ensemble number
    n_legacy_spectra = 6 #sxx, syy, szz, sxy, sxz, syz
    n_extended_spectra = 8 #sxx, syy, szz, snn, sxy, sxz, syz, szn

    def detect_spc_layout(path):
        # Detect spectral layout from actual data rows.
        import csv
        max_columns = 0
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                if len(row) > max_columns:
                    max_columns = len(row)
                if idx >= 200:
                    break
        if max_columns >= n_header_columns + 2 * n_extended_spectra * nf: #factor of 2 for imaginary part of spectra
            return 'extended'
        if max_columns >= n_header_columns + 2 * n_legacy_spectra * nf:
            return 'legacy'
        else:
            print(f"WARNING: unknown spectral layout in file {path}")
        return None

    layout = detect_spc_layout(inputFileName)
    if layout == 'extended':
        startColumnNumber = extendedStartColumnNumber
        num_spectra = n_extended_spectra # auto-, co-, and cross-spectra for four channels (x, y, z, and n)
    elif layout == 'legacy':
        startColumnNumber = legacyStartColumnNumber
        num_spectra = n_legacy_spectra # auto-, co-, and cross-spectra for three channels (x, y, and z)
    elif versionNumber in [0,2,3]: #default to version number for identifying start column number
        startColumnNumber = legacyStartColumnNumber
        num_spectra = n_legacy_spectra
    else:
        startColumnNumber = extendedStartColumnNumber
        num_spectra = n_extended_spectra

    stride = 2 * num_spectra # SPC files hold both real and imaginary parts of spectra, so stride is 2x number of spectra

    if not include_n_channel:
        gated_n_channels = {'Snn', 'Czn'}
        filtered = []
        for key in outputSpectra:
            if key in gated_n_channels:
                print(f"WARNING: skipping {key} because include_n_channel is disabled.")
                continue
            filtered.append(key)
        outputSpectra = filtered

    available_output = set(startColumnNumber.keys()) | {'a1', 'a2', 'b1', 'b2'}
    filtered = []
    for key in outputSpectra:
        if key not in available_output:
            print(f"WARNING: requested spectrum {key} is not available in this file format/version; skipping.")
            continue
        filtered.append(key)
    outputSpectra = filtered

    # Read csv file using Pandas - this is the only section in the code
    # still reliant on Pandas, and only there due to supposed performance
    # benefits.
    # Provide explicit names so mixed-width spectral rows can be parsed when
    # a file contains both legacy and extended layouts.
    tmp = pd.read_csv(
        inputFileName,
        index_col=False,
        skiprows=[0],
        header=None,
        names=list(range(5 + stride * nf)),
        usecols=tuple(range(2, 5 + stride * nf)),
    )
    
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

    # Filter lf-noise
    if lf_filter:
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
                mat = data[key]
                savemat(
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
    # match the Spotter output filename signature
    #

    if versionFileList is not None:
        #
        # Only add filenames that are in the present version file number list
        #
        fileNames = []
        for filename in sorted(os.listdir(path)):
            #
            if (fnmatch.fnmatch(filename, '????_' + suffix + '.CSV') or
                fnmatch.fnmatch(filename, '????_' + suffix + '.csv') or
                fnmatch.fnmatch(filename, '????_' + suffix + '.log')):
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
                       if
                       (fnmatch.fnmatch(filename, '????_' + suffix + '.CSV') or
                        fnmatch.fnmatch(filename, '????_' + suffix + '.csv') or
                        fnmatch.fnmatch(filename, '????_' + suffix + '.log'))]
    #
    # Are there valid Spotter files?
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
        raise ValueError('Unknown outputFileType; options are: numpy , matlab , pickle , csv')
        #
    #
#
class missingFLTFile(Exception):
    pass

def cat( path = None, outputFileName = 'displacement.CSV', Suffix='FLT',
             reportProgress=True, outputFileType='CSV',versionFileList=None, compatibilityVersion=defaultVersion):
    """
    This functions concatenates raw csv files with a header. Only for the first 
    file it retains the header. Note that for SPEC files special
    processing is done.
    """
    import gzip
    #

    def modeDetection( input_path , filename ):
        #
        # Here we detect if we are in debug or in production mode, we do this
        # based on the first few lines; in debug these will contain either
        # FFT or SPEC, whereas in production only SPECA is encountered
        mode = 'production'
        with open(os.path.join( input_path,filename) ) as infile:
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
        # Function to find nth and nth-1 occurrences of a needle in the haystack
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
        return False
    
    fname,ext = os.path.splitext(outputFileName)
    outputFileName = fname + '.' + extensions(outputFileType)
    #
    compress = outputFileType.lower()=='gz'

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
                except Exception as e:
                    message = "- ERROR:, file " + os.path.join( path,filename) + " is corrupt"
                    log_errors(message)
                    print(message )
            #end with
            #
        else:
            #
            # Suffix is not 'SPC'
            #
            fqfn = os.path.join(path, filename)
            with open(fqfn) as infile:
                #
                line_num = 0
                try:
                    for line in infile:
                        line_num += 1
                        # skip any file that doesn't end with '\n'
                        # https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
                        if not line.endswith('\n'):
                            log_errors(f"DEBUG: cat(): not SPC: skipping line {line_num} of file {filename}")
                            continue
                        # if this is the first file of this type, keep the header
                        # otherwise, drop it
                        if index > 0 and line_num == 1:
                            continue
                        if compress:
                            line = line.encode('utf-8')
                        else:
                            # Strip dos newline char
                            line = line.replace('\r', '')
                        outfile.write(line)
                except UnicodeDecodeError as e:
                    print(f"WARNING: likely corrupt line near line number {line_num} in {fqfn}, continuing: {e}")
                    continue
                # end try
            # end with
        # end if
    outfile.close()
    return True
#end def

    
def validCommandLineArgument( arg ):
    #
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
                    sha = line.split(':')
                    sha = sha[-1].strip()
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
        #
        version[-1]['fileNumbers'].append( entry )                        
        #
    #end for filenames
    #
    return version
#
def filterSOS(versionNumber, IIRWeightType):
    """
    Get second-order sections coefficients for IIR filter based on version and weight type.
    
    Args:
        versionNumber (int): Version number of the data format
        IIRWeightType (int): Type of IIR weight (0=Type A, 1=Type B, 2=Type C)
        
    Returns:
        numpy.ndarray or float: SOS coefficients matrix or 0 if unsupported
        
    Raises:
        ValueError: If IIRWeightType is not valid for the given version
        TypeError: If inputs are not numeric
    """
    
    # Input validation
    try:
        versionNumber = int(versionNumber)
        IIRWeightType = int(IIRWeightType)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Version number and IIR weight type must be numeric. Got {type(versionNumber)}, {type(IIRWeightType)}")
    
    # Version 0 and below are not supported
    if versionNumber < 1:
        return 0.0
    
    # Versions 1-3 are supported
    if versionNumber in [1, 2, 3]:
        # Define filter coefficients for each type
        filter_configs = {
            0: {  # Type A
                'lp': {'a1': -1.8514229621, 'a2': 0.8578089736, 'b0': 0.8972684452, 'b1': -1.7945369122, 'b2': 0.8972684291},
                'hp': {'a1': -1.9318795385, 'a2': 0.9385430645, 'b0': 1.0000000000, 'b1': -1.9999999768, 'b2': 1.0000000180}
            },
            1: {  # Type B
                'lp': {'a1': 1.9999999964, 'a2': 0.9999999964, 'b0': 0.9430391609, 'b1': -1.8860783217, 'b2': 0.9430391609},
                'hp': {'a1': -1.8828311523, 'a2': 0.8893254984, 'b0': 1.0000000000, 'b1': 2.0000000000, 'b2': 1.0000000000}
            },
            2: {  # Type C
                'lp': {'a1': 1.1375322034, 'a2': 0.4141775928, 'b0': 0.6012434213, 'b1': -1.2024868427, 'b2': 0.6012434213},
                'hp': {'a1': -1.8827396569, 'a2': 0.8894088696, 'b0': 1.0000000000, 'b1': 2.0000000000, 'b2': 1.0000000000}
            }
        }
        
        if IIRWeightType not in filter_configs:
            raise ValueError(f"IIRWeightType {IIRWeightType} not supported for version {versionNumber}. Valid types: {list(filter_configs.keys())}")
        
        config = filter_configs[IIRWeightType]
        lp, hp = config['lp'], config['hp']
        
        try:
            # Build SOS matrix
            sos = np.array([
                [lp['b0'], lp['b1'], lp['b2'], 1.0, lp['a1'], lp['a2']],
                [hp['b0'], hp['b1'], hp['b2'], 1.0, hp['a1'], hp['a2']]
            ])
            return sos
        except Exception as e:
            raise RuntimeError(f"Failed to create SOS matrix: {e}")
    
    else:
        raise ValueError(f"Version {versionNumber} is not supported. Supported versions: 1, 2, 3")


def applyfilter( data , kind , versionNumber, IIRWeightType ):
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


def cli_main():
    """Command-line interface for the Spotter data parser."""
    parser = argparse.ArgumentParser(
        description="Parse and concatenate Spotter SD card data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sd_file_parser.py /path/to/spotter/data
  python sd_file_parser.py /path/to/data --output_path /path/to/results
  python sd_file_parser.py /path/to/data --spectra all --format matlab
        """
    )
    
    parser.add_argument('input_path', 
                       help='Path to directory containing Spotter data files')
    parser.add_argument('--output_path', '-o', 
                       help='Output directory (default: input_path/processed)')
    parser.add_argument('--format', choices=['CSV', 'matlab', 'numpy'], 
                       default='CSV', help='Output file format')
    parser.add_argument('--spectra', default='all',
                       help='Spectra to process (e.g., Szz, all, or list)')
    parser.add_argument('--include_n_channel', action='store_true',
                       help='Include n displacement/spectral channels when present in raw files')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(args.input_path, 'processed')
    
    # Call the actual processing function
    parse_kwargs = dict(
        input_path=args.input_path,
        output_path=args.output_path,
        output_format=args.format,
        spectra=args.spectra,
    )
    if args.include_n_channel:
        parse_kwargs['include_n_channel'] = True

    parse_spotter_files(**parse_kwargs)

if __name__ == '__main__':
    cli_main()
