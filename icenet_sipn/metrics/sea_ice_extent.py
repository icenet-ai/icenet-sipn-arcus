import numpy as np
import pandas as pd
import xarray as xr

from abc import ABC

class SeaIceExtent(ABC):
    """Monthly Sea Ice Extent computation.
    
    Refer here: https://www.arcus.org/sipn/sea-ice-outlook/2023/june
    """

    def compute_sea_ice_extent(self, sea_ice_concentration, ensemble_axis=None, grid_cell_area=25*25,
                                threshold=0.15, plot=False):
        """Compute Sea Ice Extent for an image for a given day.

        Computes the total extent (SIC>15%) for one day.
        """
        sic = sea_ice_concentration
        # print(sic.data.shape)

        # Mask values less than the threshold
        sic = sic.where(sic>threshold)

        if plot:
            xr.plot.imshow(sic.squeeze())

        # Multiply by grid-cell area
        sic *= grid_cell_area

        if ensemble_axis == None or len(sic.shape) < 4:
            valid_axes = tuple(i for i in range(len(sic.shape)))
        else:
            valid_axes = tuple(i for i in range(len(sic.shape)) if i != ensemble_axis)

        # Divide by 10^6 to get:
        # (units in 10^6 km^2)
        # Sum across all axes except ensemble
        sea_ice_extent = sic.sum(axis=valid_axes) / 1E6

        return sea_ice_extent

    def compute_daily_sea_ice_extent(self, grid_cell_area=25*25, threshold=0.15, plot=False):
        """Daily Sea Ice Extent.

        Computes the total extent (SIC>15%) for each day.
        """
        kwargs = {"ensemble_axis": 0, "grid_cell_area": grid_cell_area, "threshold": threshold, "plot": plot}
        sea_ice_extent_daily_ensemble = np.asarray([self.xarr.sic.isel(leadtime=day-1).map_blocks(self.compute_sea_ice_extent, kwargs=kwargs).values for day in self.xarr.leadtime]);
        sea_ice_extent_daily_mean = sea_ice_extent_daily_ensemble.mean(axis=1)
        sea_ice_extent_daily_stddev = sea_ice_extent_daily_ensemble.std(axis=1)
        forecast_dates = pd.to_datetime(self.xarr.forecast_date)
        sea_ice_extent_daily_ds = xr.Dataset(
            data_vars=dict(
                sea_ice_extent_daily = (["day", "ensemble"], sea_ice_extent_daily_ensemble),
                sea_ice_extent_daily_mean = (["day"], sea_ice_extent_daily_mean),
                sea_ice_extent_daily_stddev = (["day"], sea_ice_extent_daily_stddev),
            ),
            coords=dict(
                day=forecast_dates,
                ensemble=list(range(sea_ice_extent_daily_ensemble.shape[1]))
            )
        )
        return sea_ice_extent_daily_ds

    def compute_monthly_sea_ice_extent(self, grid_cell_area=25*25, threshold=0.15, plot=False):
        """Monthly Sea Ice Extent from daily.

        Computes the total extent (SIC>15%) for each day, then, averages the extent
        from each of the days of the month into a monthly average extent.

        To be consistent with NSIDC Sea Ice Index extent.
        """
        sea_ice_extent_daily_ds = self.compute_daily_sea_ice_extent(grid_cell_area=grid_cell_area, threshold=threshold, plot=plot)

        sea_ice_extent_monthly_ds = sea_ice_extent_daily_ds.sea_ice_extent_daily_mean.groupby(sea_ice_extent_daily_ds.day.dt.month).mean().rename("sea_ice_extent_monthly_mean")

        self.xarr["sea_ice_extent_monthly_mean"] = sea_ice_extent_monthly_ds

        return sea_ice_extent_monthly_ds

    def get_data(self):
        return self.xarr
