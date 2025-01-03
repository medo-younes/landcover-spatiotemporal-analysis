import rasterio
import rioxarray as rio
import os
import rasterio
from dask.distributed import Client, LocalCluster, Lock
import xarray as xr
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from rasterio.enums import Resampling
from shapely.geometry import shape
import geopandas as gpd
from rasterio.features import shapes

def classify_ndvi(ndvi, min_threshold=0.19, bins=[0.2, 0.4, 0.6, 1.0]):
    '''
    Classifies an NDVI Raster band into a classes falling into a set of ranges

    0.1 or less = Areas of barren rock, sand, or snow usually show very low NDVI values
    0.2 to 0.5 = Sparse vegetation such as shrubs and grasslands or senescing crops may result in moderate NDVI values 
    0.6 to 1.0 = High NDVI values correspond to dense vegetation such as that found in temperate and tropical forests or crops at their peak growth stage.     
    '''
    # Classify using digitize
    ndvi=ndvi.where(ndvi > min_threshold)
    binned = xr.DataArray(np.digitize(ndvi, bins=bins), dims=ndvi.dims, coords=ndvi.coords)
  
    binned=binned.where(binned!= binned.max())
    return binned


def xr_to_train(raster,bands=None,mask=True):


    if bands != None:
        raster=raster.sel(band=bands)
    

    raster=raster.where(raster != raster.attrs["_FillValue"])

    # Exract and Reshape Values
    vals=raster.transpose('y', 'x', 'band').values
    shape=vals.shape
    data=vals.reshape((shape[0] * shape[1], shape[2]))

    if mask:
    # Create a masked array of non-NaN values
        return np.ma.masked_invalid(data)
        # valid_data=masked.data[~masked.mask.any(axis=1),:]
        
    else:
        return data

def kmeans_raster(raster,bands, scale=True,k=3,random_state=42):
    
    orig_shape=raster.shape
    X = xr_to_train(raster, bands=bands, mask=True)
    X_mask=X.data[~X.mask.any(axis=1),:]

    if scale:
        # # Standardize the data (optional but recommended for clustering)
        scaler = StandardScaler()
        X_mask = scaler.fit_transform(X_mask)

    # Run K-Means clustering 3n the provided bands
    kmeans=KMeans(n_clusters=k, random_state=random_state)
    clusters=kmeans.fit_predict(X_mask)

    # Strucutre Cluster Values into a XR DataArray
    cluster_map=np.full(X.shape[0],np.nan)
    cluster_map[~X.mask.any(axis=1)] = clusters
    cluster_raster=cluster_map.reshape((orig_shape[1], orig_shape[2]))
    cluster_da=raster.isel(band=0)
    cluster_da.values=cluster_raster

    # return cluster_da
    return cluster_da

def kmeans_df(df, k =5):

    '''

    KMeans clustering on a Pandas DataFrame end-to-end
    

    Parameters
    ----------
    df: dataframe
    Pandas DataFrame of values to cluster, in this case typically values of raster reshaped to (width * height, bands)
    
    k : int
    Number of clusters for KMeans

    Outputs
    ---------

    labels: list
    List of cluster labels with length = len(df)
    
    '''

    scaler=StandardScaler() # Initialize StandardScaler 
    df_scaled=scaler.fit_transform(df) # Rescale df values to 

    kmeans=KMeans(n_clusters=k,random_state=42) # Initalize KMeans model with n_clusters=k

    return kmeans.fit_predict(df_scaled) # Predict cluster labels for scaled df



def ndvi(nir, red):
    '''Calculate NDVI from integer arrays'''
    ndvi = (nir - red) / (nir + red) # This line calculates the NDVI using the formula: (NIR - Red) / (NIR + Red).
    return ndvi



def ndbi(swir,nir):
    '''Calculate NDBI from integer arrays'''
    ndbi = (swir - nir) / (swir + nir)
    return ndbi



def ndwi(nir,swir):
    '''Calculate NDWI from integer arrays'''
    ndwi = (nir - swir) / (nir + swir)
    return ndwi


# Define a function
def nbr(nir, swir):
    '''Calculate NBR using NIR and SWIR bands.'''
    nbr = (nir - swir) / (nir + swir)
    return nbr


def tci(tir):
    tir=tir.where(tir!=tir.attrs["_FillValue"])
    return 100 * (tir.max() -  tir) / (tir.max() - tir.min())


def lc_change(lc_trend, time_scale="Week"):

    '''

    Resample an xarray DataArray to a targett resolution
    

    Parameters
    ----------
    lc_trend: list
    1-D array of land cover status over time, each index is a subsequent time stamp
    
    time_scale: str
    For output labels to show the time stage at which the land cover change occured

    Outputs
    ---------

    dict(
        lc_start=lc_start, # Starting Land Cover Class
        lc_middle=lc_middle, # Land Cover Class after change occured
        lc_final=lc_final, # Final Land Cover Class, may be different to middle land cover class
        time_change=time_change, # Time at which the land cover change occured
        label_name=label_name # User friendly string label indicating The Land Cover Change that occured
    )
    
    '''

    mask=[lc_trend[i] == lc_trend[i+1] for i in range(0, len(lc_trend)-1)]
    mask.insert(0,True)
    
    try:
        change_idx=mask.index(False)
    except:
        change_idx= -1

    lc_start=lc_trend[0]
    lc_middle=lc_trend[change_idx]
    lc_final=lc_trend[-1]
    time_change=change_idx+1

    if lc_start == lc_middle or lc_start == lc_final:
        label_name= "No Change"
    else:
        label_name=f"{lc_start.title()} to {lc_middle.title()} on {time_scale} {change_idx + 1}"

    return dict(
        lc_start=lc_start,
        lc_middle=lc_middle,
        lc_final=lc_final,
        time_change=time_change,
        label_name=label_name
    )

     
def resample_da(raster,target_res):

    '''

    Resample an xarray DataArray to a targett resolution
    

    Parameters
    ----------
    raster: xr.DataArray
    Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
    
    target_res : int
    Cell resolution in METERS that you want to resample the raster to

    Outputs
    ---------

    rescaled: xr.DataArray
    Raster rescaled to the target resolution, keeping the original projection of the input raster
    
    '''
    if target_res > 0.0099 and target_res <= 0.99:
        print(f"Resampling input raster to {target_res * 100} cm resolution")
    elif  target_res <= 0.0099:
        print(f"Resampling input raster to {target_res * 1000} mm resolution")
    else:
        print(f"Resampling input raster to {target_res} m resolution")

    orig_res=raster.rio.resolution() # Get original resolution of raster
    orig_res_x = orig_res[0] # resolution in x / lon direction
    orig_res_y = abs(orig_res[1]) # resolution in y / lat direction

    orig_width = raster.rio.width # original width
    orig_height = raster.rio.height # original height


    rescale_x=target_res / orig_res_x # X rescale factor
    rescale_y=target_res / orig_res_y # Y rescale factor

    target_width= round(orig_width / rescale_x) # Calculate Target Width
    target_height=round(orig_height / rescale_y) # Calculate Target Height

    # Reample input raster to the new dimensions
    ## Bilinear resampling set as default resampling method
    resampled = raster.rio.reproject(
        raster.rio.crs,
        shape=(target_height, target_width),
        resampling=Resampling.bilinear,
    )

    return resampled


def xr_vectorize(raster,pixel_threshold=2, col_name="value"):
    '''

    Resample an xarray DataArray to a targett resolution
    

    Parameters
    ----------
    raster: xr.DataArray
    Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
    
    target_res : int
    Cell resolution in METERS that you want to resample the raster to

    Outputs
    ---------

    rescaled: xr.DataArray
    Raster rescaled to the target resolution, keeping the original projection of the input raster
    
    '''
    
    resolution=raster.rio.resolution()
    cell_area_m2= resolution[0] * abs(resolution[1])
    mask=raster.values != np.nan

    raster_values=raster.astype(int).values

    # Extract geometries and values
    shapes_generator = shapes(raster_values, mask=mask, transform=raster.rio.transform())
    

    raster_shapes=[[shape(geom),value] for geom, value in shapes_generator if value != np.nan]
    raster_shapes=np.array(raster_shapes).transpose()


    gdf = gpd.GeoDataFrame({col_name: raster_shapes[1]}, geometry= raster_shapes[0], crs=raster.rio.crs)

    gdf["area_m2"]=gdf.area 
    gdf["pixels"]=gdf.area_m2 / cell_area_m2
    gdf=gdf[gdf.pixels > pixel_threshold]

    return gdf.astype({"class":int})
 

def get_raster_class(raster,val,range, fill_val=0):
    raster=raster.where((raster >= range[0])  & (raster < range[1])) 
    return raster.where(raster.isnull(),val).where(~raster.isnull(),fill_val)