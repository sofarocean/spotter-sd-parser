import pandas as pd
import xarray as xr
import os
import numpy as np

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
    
    #TODO - rename variables and set metadata according to CF Conventions
    
    spot.to_netcdf(outpath)

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