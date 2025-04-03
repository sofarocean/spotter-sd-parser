import pandas as pd
import xarray as xr
import os
import numpy as np
from datetime import datetime
import getpass

def main(path=None, outpath=None):
    
    if path is None:
        path = os.getcwd()
    else:
        path = os.path.abspath(path)
    
    if outpath is None:
        outpath = os.path.join(os.getcwd(), 'spotter.nc')
    else:
        outpath = os.path.abspath(outpath)
        
    # Read CSVs to Pandas DataFrames    
    infiles = ['Szz', 'Sxx', 'Syy', 'Qxz', 'Qyz', 'Cxy', 'a1', 'b1', 
               'a2', 'b2', 'location', 
            #    'displacement', #TODO - this is slow for some reason...?
               'bulkparameters']

    dfs = {}
    def date_parser(x):
        return pd.to_datetime(x.strip(), format="%Y %m %d %H %M %S %f")

    for file in infiles:
        file_to_read = os.path.join(path,file + '.csv')
        print(file_to_read) #TODO
        dfi = pd.read_csv(file_to_read,
                            index_col=0,
                            parse_dates=[np.arange(7).tolist()],
                            date_parser=date_parser)
        dfi.index.name = 'time'
        dfi.columns = dfi.columns.str.strip()
        dfs[file] = dfi
        
    # Convert DataFrames to Xarray Datasets/DataArrays, merge
    ds_list = []
    for file in infiles[:10]:
        dai = dfs[file].iloc[:, 1:].to_xarray().to_array(dim='freq', name=file)
        dai["freq"] = dai["freq"].astype('float')
        dai = dai.astype('float')
        ds_list.append(dai)
        
    for file in infiles[10:]:
        dsi = dfs[file].to_xarray()
        ds_list.append(dsi)
        
    spot = xr.merge(ds_list)
    
    spot = spot.rename({
        'Significant Wave Height':'Hm0',
        'Mean Period': 'Tm',
        'Peak Period': 'Tp',
        'Mean Direction':'Dm',
        'Peak Direction': 'Dp',
        'latitude (decimal degrees)': 'lat',
        'longitude (decimal degrees)': 'lon'
    })
    
    spot['Szz'].attrs['units'] = 'm^2/Hz'
    spot['Szz'].attrs['standard_name'] = 'sea_surface_wave_variance_spectral_density'
    spot['Szz'].attrs['long_name'] = 'Spectral density'
    
    spot['Sxx'].attrs['units'] = 'm^2/Hz'
    #TODO spot['Sxx'].attrs['standard_name'] = ''
    spot['Sxx'].attrs['long_name'] = 'Variance density spectra of eastward displacement'
    
    spot['Syy'].attrs['units'] = 'm^2/Hz'
    #TODO spot['Syy'].attrs['standard_name'] = ''
    spot['Syy'].attrs['long_name'] = 'Variance density spectra of northward displacement'
    
    spot['Qxz'].attrs['units'] = 'm^2/Hz'
    #TODO spot['Qxz'].attrs['standard_name'] = ''
    spot['Qxz'].attrs['long_name'] = 'Quad-spectrum between vertical and eastward displacement'
    
    spot['Qyz'].attrs['units'] = 'm^2/Hz'
    #TODO spot['Qyz'].attrs['standard_name'] = ''
    spot['Qyz'].attrs['long_name'] = 'Quad-spectrum between vertical and northward displacement'
    
    spot['Cxy'].attrs['units'] = 'm^2/Hz'
    #TODO spot['Cxy'].attrs['standard_name'] = ''
    spot['Cxy'].attrs['long_name'] = 'Co-spectrum between northward and eastward displacement'
    
    spot['a1'].attrs['units'] = ' '
    #TODO spot['a1'].attrs['standard_name'] = ''
    spot['a1'].attrs['long_name'] = 'First order cosine coefficient'
    
    spot['b1'].attrs['units'] = ' '
    #TODO spot['b1'].attrs['standard_name'] = ''
    spot['b1'].attrs['long_name'] = 'First order sine coefficient'
    
    spot['a2'].attrs['units'] = ' '
    #TODO spot['a2'].attrs['standard_name'] = ''
    spot['a2'].attrs['long_name'] = 'Second order cosine coefficient'
    
    spot['b2'].attrs['units'] = ' '
    #TODO spot['b2'].attrs['standard_name'] = ''
    spot['b2'].attrs['long_name'] = 'Second order sine coefficient'
    
    spot['Hm0'].attrs['units'] = 'm'
    spot['Hm0'].attrs['standard_name'] = 'sea_surface_wave_significant_height'
    spot['Hm0'].attrs['long_name'] = 'Significant wave height'
    
    spot['Tm'].attrs['units'] = 's'
    spot['Tm'].attrs['standard_name'] = 'sea_surface_wave_zero_upcrossing_period'
    spot['Tm'].attrs['long_name'] = 'Mean period'
    
    spot['Tp'].attrs['units'] = 's'
    spot['Tp'].attrs['standard_name'] = 'sea_surface_wave_period_at_variance_spectral_density_maximum'
    spot['Tp'].attrs['long_name'] = 'Peak period'
    
    spot['Dm'].attrs['units'] = 'degree'
    spot['Dm'].attrs['standard_name'] = 'sea_surface_wave_from_direction'
    spot['Dm'].attrs['long_name'] = 'Mean direction'
    
    spot['Dp'].attrs['units'] = 'degree'
    spot['Dp'].attrs['standard_name'] = 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum'
    spot['Dp'].attrs['long_name'] = 'Peak direction'
    
    spot['lat'].attrs['units'] = 'degree_north'
    spot['lat'].attrs['standard_name'] = 'latitude'
    spot['lat'].attrs['long_name'] = 'Latitude'

    spot['lon'].attrs['units'] = 'degree_east'
    spot['lon'].attrs['standard_name'] = 'longitude'
    spot['lon'].attrs['long_name'] = 'Longitude'
    
    spot['freq'].attrs['units'] = 'Hz'
    #TODO spot['freq'].attrs['standard_name'] = ''
    spot['freq'].attrs['long_name'] = 'Frequency'
    
    #TODO spot['time'].attrs['units'] = 'Hz'
    #TODO spot['time'].attrs['standard_name'] = ''
    spot['time'].attrs['long_name'] = 'Time'

    spot.attrs['Conventions'] = 'CF-1.8'
    spot.attrs['source'] = 'Sofar spotter buoy'
    spot.attrs['history'] = 'generated {:} by {:}'.format(datetime.now().strftime('%Y-%m-%d @ %H:%M:%S'),
                                    getpass.getuser())
    spot.attrs['references'] = 'https://content.sofarocean.com/hubfs/Technical_Reference_Manual.pdf'
    
    spot.to_netcdf(os.path.join(outpath, 'spot.nc'))
    return spot

if __name__ == "__main__":
    # # TODO - allow for options
    # # execute only if run as a script
    # #
    # import sys

    # narg      = len( sys.argv[1:] )
    
    # if narg>0:
    #     #
    #     #parse and check command line arguments
    #     arguments = dict()
    #     for argument in sys.argv[1:]:
    #         #
    #         key,val = validCommandLineArgument( argument )
    #         arguments[key]=val            
    #         #
    #     #
    # else:
    #     #
    #     arguments = dict()
    #     #
    # #
    main()