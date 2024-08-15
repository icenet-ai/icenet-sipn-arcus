import logging
import os

from abc import ABC
from glob import glob
from pathlib import Path

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

from icenet.plotting.video import xarray_to_video as xvid
from icenet.data.sic.mask import Masks
from icenet.process.predict import get_prediction_data


class IceNetOutputPostProcess(ABC):
    def __init__(self, prediction_pipeline_path: str,
                 prediction_name: str, date: dt.date) -> None:
        self.prediction_pipeline_path = prediction_pipeline_path
        self.prediction_name = prediction_name
        self.date = date


    @property
    def get_hemisphere(self):
        """
        Return the hemisphere based on the attributes `north` and `south` of the IceNetOutputPostProcess class instance.
        If `north` is True, return string: "north".
        If `south` is True, return string: "south".
        Otherwise, raise an Exception indicating that the hemisphere is not specified.
        """
        if self.north:
            return "north"
        elif self.south:
            return "south"
        else:
            raise Exception("Hemisphere not specified!")


    @property
    def get_pole(self):
        """
        Return the pole value based on the attributes 'north' and 'south' of the IceNetOutputPostProcess class instance.
        If `north` is True, return 1.
        If `south` is True, return -1.
        Otherwise, raise an Exception indicating that the hemisphere is not specified.
        """
        if self.north:
            return 1
        elif self.south:
            return -1
        else:
            raise Exception("Hemisphere not specified!")


    def get_mask(self):
        """
        Generate the land mask using the Masks class instance with the specified parameters.
        Returns the Masks class instance and the generated land mask.
        """
        mask_gen = Masks(south=self.south, north=self.north)
        land_mask = mask_gen.get_land_mask()
        return mask_gen, land_mask


    def create_ensemble_dataset(self, date_index: bool = False):
        """
        Create an xarray Dataset including ensemble prediction data based on netCDF
        output by icenet.

        Args:
            date_index: If True, set forecast dates as index; otherwise, use
                forecast index integers.

        Returns:
            xarray.Dataset: Dataset containing ensemble prediction data.
        """

        icenet_output_netcdf_file = os.path.join(self.prediction_pipeline_path, "results",
                                            "predict", self.prediction_name) + ".nc"
        ds = xr.open_dataset(icenet_output_netcdf_file)

        # Dimensions for all data being inserted into xarray
        if date_index:
            data_dims_list = ["ensemble", "time", "yc", "xc", "forecast_date"]
        else:
            data_dims_list = ["ensemble", "time", "yc", "xc", "leadtime"]

        # Apply np.nan to land mask regions (emulating icenet output)
        mask_gen, land_mask = self.get_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[land_mask] = np.nan
        land_mask_nan[~land_mask] = 1.0


        arr, data, ens_members = get_prediction_data(
            root=self.prediction_pipeline_path,
            name=self.prediction_name,
            date=self.date,
            return_ensemble_data=True,
        )
        dates = pd.to_datetime(ds.forecast_date[0])

        # Apply 0.0 to inactive cell regions
        for i, forecast_lead_date in enumerate(dates):
            # Get active cell mask for month of this forecast lead date
            grid_cell_mask = mask_gen.get_active_cell_mask(forecast_lead_date.month)
            # Applying to SIC mean read from numpy to directly compare against
            # `icenet_output` = 0
            # Apply to SIC mean
            arr[~grid_cell_mask, i, 0] = 0.0
            # Apply to SIC std dev
            arr[~grid_cell_mask, i, 1] = 0.0

            # Applying to Ensemble prediction outputs
            for ensemble in range(ens_members):
                data[ensemble, :, ~grid_cell_mask, i] = 0.0

        xarr = ds.copy()

        # Add full ensemble prediction data to the original icenet DataSet
        xarr[f"sic"] = (data_dims_list, data*land_mask_nan[np.newaxis, np.newaxis, :, :, np.newaxis])

        self.ds = ds

        if not date_index:
            self.xarr = xarr
        else:
            return xarr

        ds.close()


    def save_data(self, output_path, reference="BAS_icenet", drop_vars=None):
        output_path = os.path.join(output_path,
                                "netcdf",
                                f"{reference}_{self.get_hemisphere}.nc"
                                )

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        if drop_vars is None:
            xarr = self.xarr
        else:
            xarr = self.xarr.copy()
            xarr = xarr.drop_vars(drop_vars, errors="ignore")
        
        compression = dict(zlib=True, complevel=9)
        vars_encoding = {var: compression for var in xarr.data_vars}
        coords_encoding = {coord: compression for coord in xarr.coords}

        xarr.to_netcdf(output_path,
                        encoding=vars_encoding | coords_encoding
                    )


    def get_data_date_indexed(self):
        """Get forecast Xarray Dataset with forecast dates set as index
        instead of forecast index integers.
        """
        xarr = self.create_ensemble_dataset(date_index=True)
        return xarr


    def plot_ensemble_mean(self):
        """Plots the ensemble mean.
        """
        self.create_ensemble_dataset()
        mask_gen, land_mask = self.get_mask()

        forecast_date = self.xarr.time.values[0]
        
        fc = self.xarr.sic_mean.isel(time=0).drop_vars("time").rename(dict(leadtime="time"))
        fc['time'] = [pd.to_datetime(forecast_date) \
                    + dt.timedelta(days=int(e)) for e in fc.time.values]

        # Convert eastings and northings from kilometers to metres.
        # Needed if coastlines is enabled in following `xvid`` call.
        fc["xc"] = fc["xc"].data*1000
        fc["yc"] = fc["yc"].data*1000

        anim = xvid(fc, 15, figsize=(8, 8), mask=land_mask, mask_type="contour", north=self.north, south=self.south, coastlines=False)
        return anim
