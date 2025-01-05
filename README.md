# ML-based Spatiotemporal Analysis of Land Cover Time Series Data

This project showcases a novel approach to analysing, extracting and visualizing significant land cover change events occuring across a time series of remote sensing imagery. The output is a polygon layer with regions dilineating the extent of change, the regions are labeled with a description indicating the land cover change that occured and the time period at which it occured.

The example data used in this project is sourced from a research paper that collected UAV multispectral imagery of Niwot Ridge, Colorado between 21 June - 14 August, 2017. The dataset highlights significant changes in snow cover and vegetation health during the summer period in a highly remote alpine ecosystem. Normalized Difference Vegetation Index (NDVI) was utilized to classify land cover into 4 distinct classes; Dense Vegetation, Moderate Vegetation, Sparse Vegetation and Snow. Resultantly, 7 land cover maps of the Niwot Ridge area were derived, each a week apart. Spatial dimensions in addition to the land cover classifications at each date (temporal dimension) were extracted into a single dataframe for KMeans clustering. The clustering algorithm distinguishes regions whereby land cover change occurs at a certain period of time. Following further analysis and post-processing of the clustered dataset, the nature of land cover change can be labeled according to a basic format such as "LC 1 to LC 2 on Date X", indicating the specific time point at which the land cover change is occuring. The output dataset can then be intuitively visualized and further analyzed for deeper insights into the underlying spatiotemporal trends.

<img src="visualization/NiwotRidge_SpatioTemporal.png" width="500" padding-right="250">


The workflow is as follows:
1. Calculate NDVI from NIR and Red Bands
2. Classify Land Cover from NDVI
3. Extract spatial coordinates and raster values from each timepoint
4. KMeans clustering of spatial coordinates and land cover values at each timepoint
5. Change Classification function outputing label description of the type of change that occured and the time
6. Vectorization of clustered raster and joining with change decsriptors
