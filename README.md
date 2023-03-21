# Spotter SD Card Data Parser
An open-source tool for concatenating and parsing SD card data from Spotters.

# Purpose:

For efficiency, Spotter stores wave spectra, mean location and 
displacement data on the SD card across multiple files. In order to 
access the data for post processing it is convenient to first 
recombine each data type in a single file. 

This module contains functions to process Spotter output files containing 
spectra, mean location and displacement information, and concatenate all 
files pertaining to a specific data type (e.g. displacements) into a single
comma delimited (CSV) file. For example, all displacement information 
(contained in `????_FLT.CSV`) is combined as:

       (input)                                (output) 
    0010_FLT.CSV  -|
    0011_FLT.CSV   |      running script
    0012_FLT.CSV   |           ==== >       displacement.CSV
    ............   |
    000N_FLT.CSV  -|

and similarly for spectral ( `xxxx_SPC.CSV` => `Szz.csv`) and location 
(`xxxx_LOC.CSV` => `location.csv`) files. Further, after all spectral files have
been combined. Bulk parameters (significant wave height, peak period, etc.) are calculated from the spectral files,
and stored seperately in `bulkparameters.csv`

NOTE: the original data files will remain unchanged.

# Installation

In order to use this script, python (version 2 or 3) needs to be installed
on the system (download at: www.python.org). In addition, for functionality 
the script requires that the following python modules:

        dependencies: pandas, numpy, scipy
         
These modules can be installed by invoking the python package manager
(pip) from the command line. For instance, to install pandas you would
run the package manager from the command line as:

        pip3 install pandas

and similarly for other missing dependencies.

# Usage

To use the module, simply copy the Spotter files and this script into the
same directory. Subsequently, start a command line terminal, navigate 
to the directory containing the files and run the python script from the 
command line using the python interpreter as:

        python3 sd_file_parser.py

or any other python interpreter (e.g. ipython, python etc.).

## Requesting additional output:

By default, the script will only produce the variance density spectrum.
If in addition the directional moments are desired, add the command line
switch `spectra=all`, i.e.:

        python3 sd_file_parser.py spectra='all'

in which case files containing a1,b1,a2,b2 (in separate files) will be
produced.

# Output

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
parameters calculated at the indicated time:

    HEADER:    # year , month , day, hour ,min, sec, milisec , Significant Wave Height, Mean Period, Peak Period, Mean Direction, Peak Direction, Mean Spreading, Peak Spreading
               2017,11   ,10 ,5   ,3  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
               2017,11   ,10 ,5   ,33 ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
                |    |    |   |    |   |   |        |     | ,   | , | , |  , |   , |  , |
               2017,12   ,20 ,0   ,6  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr

For the definitions used to calculate the bulk parameters from the variance density spectra, 
and a short description please refer to:
https://content.sofarocean.com/hubfs/Spotter%20product%20documentation%20page/wave-parameter-definitions.pdf


# History of major updates

    Author   | Date      | Firmware Version | Script updates
    -----------------------------------------------------------------------
    P.B.Smit | Feb, 2018 | 1.4.2            | firmware SHA verification
    P.B.Smit | May, 2018 | 1.5.1            | Included IIR phase correction
    P.B.Smit | June, 2019| 1.7.0            | Bulk parameter output
    P.B.Smit | Oct, 2019 | 1.8.0            | SST Spotter update
    various  | Dec, 2021 | 1.8.0+, 2.0.0+   | Spotter v3 update

# Contributing

We encourage a standard [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow):  please create a branch, and submit a pull request when ready.  Thanks in advance.

# License

Apache 2.0.  See [LICENSE](../main/LICENSE)
