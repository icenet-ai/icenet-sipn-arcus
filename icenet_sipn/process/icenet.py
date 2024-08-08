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


class IceNetOutputPostProcess(ABC):
    def __init__(self, prediction_path, date: dt.date) -> None:
        self.prediction_path = prediction_path
        self.date = date

    @property
    def get_hemisphere(self):
        if self.north:
            return "north"
        elif self.south:
            return "south"
        else:
            raise Exception("Hemisphere not specified!")

    @property
    def get_pole(self):
        if self.north:
            return 1
        elif self.south:
            return -1
        else:
            raise Exception("Hemisphere not specified!")


    def get_mask(self):
        mask_gen = Masks(south=self.south, north=self.north)
        land_mask = mask_gen.get_land_mask()
        return mask_gen, land_mask

    def get_prediction_data_nb(self) -> tuple:
        """Read individual ensemble outputs of IceNet prediction.
        Based on IceNet library v0.2.7.
        
        Ref:
        https://github.com/icenet-ai/icenet/blob/cb1cb785808ba1138c0ed0f8f88208a144daa6ff/icenet/process/predict.py#L63-L89
        """
        glob_str = os.path.join(self.prediction_path, "*",
                                self.date.strftime("%Y_%m_%d.npy"))

        np_files = glob(glob_str)
        if not len(np_files):
            logging.warning("No files found")
            return None

        data = [np.load(f) for f in np_files]
        data = np.array(data)
        ens_members = data.shape[0]

        logging.debug("Data read from disk: {} from: {}".format(
            data.shape, np_files))

        return np.stack([data.mean(axis=0), data.std(axis=0)],
                        axis=-1).squeeze(), data, ens_members


    def create_ensemble_dataset(self, date_index=False):
        ds = xr.open_dataset(f"{self.prediction_path}.nc")

        # Dimensions for all data being inserted into xarray
        if date_index:
            data_dims_list = ["ensemble", "time", "yc", "xc", "forecast_date"]
        else:
            data_dims_list = ["ensemble", "time", "yc", "xc", "leadtime"]

        # Dict to store all ensemble results (for creating an xarray)
        sic_data = {}

        # Apply np.nan to land mask regions (emulating icenet output)
        mask_gen, land_mask = self.get_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[land_mask] = np.nan
        land_mask_nan[~land_mask] = 1.0

        arr, data, ens_members = self.get_prediction_data_nb()
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

        # for ensemble in range(ens_members):
        sic_data[f"sic"] = (data_dims_list, data*land_mask_nan[np.newaxis, np.newaxis, :, :, np.newaxis])

        sic_data.keys()

        sic_mean = ds.sic_mean.data
        logging.info("SIC Mean shape: ", sic_mean.shape, "Grid Cell Mask Shape: ", grid_cell_mask.shape)
        # sic_mean[:, ~grid_cell_mask, ...] = np.nan
        sic_stddev = ds.sic_stddev.data
        # sic_stddev[:, ~grid_cell_mask, ...] = np.nan

        # Create a dict with all variables to be stored in the xarray DataSet (incl. above ensemble results)
        data_vars=dict(
            Lambert_Azimuthal_Grid=ds.Lambert_Azimuthal_Grid,
            sic_mean=(data_dims_list[1:], sic_mean),
            sic_stddev=(data_dims_list[1:], sic_stddev),
            ensemble_members=(ens_members),
        )

        data_vars.update(sic_data)

        data_vars.keys()

        # Update longitude range to be 0 to 360 instead of -180 to 180
        # This would be for SIPN South submission, not SIPN Arcus.
        longitude = ds.lon.data.copy()
        ## longitude[longitude<0] += 360

        # print(ds.time.data)
        xarr = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                ensemble=list(range(ens_members)),
                time=[ds.time[0].data],
                leadtime=ds.leadtime.data,
                forecast_date=ds.forecast_date[0].data,
                xc=ds.xc.data,
                yc=ds.yc.data,
                lat=(("yc", "xc"), ds.lat.data),
                lon=(("yc", "xc"), longitude),
            ),
            attrs=dict(
                Conventions="CF-1.6 ACDD-1.3",
                creator_email="bryald@bas.ac.uk",
                creator_institution="British Antarctic Survey",
                creator_name="Bryn Noel Ubald",
                creator_url="www.bas.ac.uk",
                date_created=dt.datetime.now().strftime("%Y-%m-%d"),
                geospatial_bounds_crs=ds.attrs["geospatial_bounds_crs"],
                geospatial_lat_min=ds.attrs["geospatial_lat_min"],
                geospatial_lat_max=ds.attrs["geospatial_lat_max"],
                geospatial_lon_min=ds.attrs["geospatial_lon_min"],
                geospatial_lon_max=ds.attrs["geospatial_lon_max"],
                geospatial_vertical_min=0.0,
                geospatial_vertical_max=0.0,
                history="{} - creation".format(dt.datetime.now()),
                id=f"{ds.attrs['id']}",
                institution="British Antarctic Survey",
                keywords=ds.attrs["keywords"],
                keywords_vocabulary="GCMD Science Keywords",
                license="Open Government Licence (OGL) V3",
                naming_authority="uk.ac.bas",
                platform="BAS HPC",
                product_version=ds.attrs["product_version"],
                project="IceNet",
                publisher_institution="British Antarctic Survey",
                source=ds.attrs["source"],
                spatial_resolution=ds.attrs["spatial_resolution"],
                # Values for any standard_name attribute must come from the CF
                # Standard Names vocabulary for the data file or product to
                #  comply with CF
                standard_name_vocabulary=ds.attrs["standard_name_vocabulary"],
                summary=ds.attrs["summary"],
                # Use ISO 8601:2004 duration format, preferably the extended format
                # as recommended in the Attribute Content Guidance section.
                time_coverage_start=ds.attrs["time_coverage_start"],
                time_coverage_end=ds.attrs["time_coverage_end"],
                time_coverage_duration=ds.attrs["time_coverage_duration"],
                time_coverage_resolution=ds.attrs["time_coverage_duration"],
                title="Sea Ice Concentration Prediction",
            )
        )

        xarr.time.attrs = ds.time.attrs
        xarr.yc.attrs = ds.yc.attrs
        xarr.xc.attrs = ds.xc.attrs
        xarr.leadtime.attrs = ds.leadtime.attrs
        xarr.lat.attrs = ds.lat.attrs
        xarr.lon.attrs = ds.lon.attrs
        xarr.sic_mean.attrs = ds.sic_mean.attrs
        xarr.sic_stddev.attrs = ds.sic_stddev.attrs
        xarr.ensemble_members.attrs = ds.ensemble_members.attrs
        xarr.sic.attrs = dict(
            long_name="sea ice area fraction across ensemble runs of icenet model",
            standard_name="sea_ice_area_fraction",
            short_name="sic",
            valid_min=0,
            valid_max=1,
            ancillary_variables="sic_stddev",
            grid_mapping="Lambert_Azimuthal_Grid",
            units="1",
        )
        xarr.ensemble.attrs = dict(
            long_name="id of ensemble runs",
        )
        xarr.forecast_date.attrs = dict(
            long_name="array of forecast dates",
        )

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
            xarr = xarr.drop_vars(drop_vars)
        
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

        anim = xvid(fc, 15, figsize=(12, 12), mask=land_mask, mask_type="contour", north=self.north, south=self.south, coastlines=False)
        return anim
