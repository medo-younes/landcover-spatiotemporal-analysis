# Spatiotemporal Analysis of Land Cover Changes

This project showcases a novel approach to analysing, extracting and visualizing significant spatiotemporal events occuring across a time series of remote sensing imagery. The output is a polygon layer with regions dilineating the extent of change, the regions are labeled with a description indicating the land cover change that occured and the time period at which it occured.

The example data used in this project is sourced from a research paper that collected UAV multispectral imagery of Niwot Ridge, Colorado between 21 June - 14 August, 2017. The dataset highlights significant changes in snow cover and vegetation health during the summer period in a highly remote alpine ecosystem.

[!alt text](visualization/NiwotRidge_SpatioTemporal.png)

The workflow is as follows:
1. Calculate NDVI from NIR and Red Bands
2. Classify Land Cover from NDVI
3. Extract spatial coordinates and raster values from each timepoint
4. KMeans clustering of spatial coordinates and land cover values at each timepoint
5. Change Classification function outputing label description of the type of change that occured and the time
6. Vectorization of clustered raster and joining with change decsriptors
