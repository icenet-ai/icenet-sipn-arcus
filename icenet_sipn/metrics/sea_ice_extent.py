import numpy as np
import pandas as pd
import xarray as xr

from ..utils import drop_variables


class SeaIceExtent:
    """Monthly Sea Ice Extent computation.
    
    Refer here: https://www.arcus.org/sipn/sea-ice-outlook/2023/june
    """

    def get_data(self):
        return self.xarr

    def clear_sie(self):
        """Drop `sea ice extent` related variables if previously saved in dataset.
        """
        variable_names = [
            "sea_ice_extent_daily",
            "sea_ice_extent_daily_mean",
            "sea_ice_extent_monthly_mean",
            "sea_ice_extent_daily_stddev",
        ]
        self.xarr = drop_variables(self.xarr, variable_names)

    def compute_sea_ice_extent(self, sea_ice_concentration, ensemble_axis=0, grid_cell_area=25*25,
                                threshold=0.15, plot=False):
        """Compute Sea Ice Extent for an image for a given day.

        Computes the total extent (SIC>15%) for one day.

        Args:
            ensemble_axis: Axis at which the ensemble members are stored.
                            Used to omit reduction operation across this axis.
        """
        self.clear_sie()
        sic = sea_ice_concentration

        # Mask values less than the threshold
        sic = xr.where(cond=sic>threshold, x=1, y=0)

        if plot:
            xr.plot.imshow(sic.squeeze())

        # Multiply by grid-cell area
        sic *= grid_cell_area

        # If dimension is 3, the mean SIC is input.
        # If dimension is 4, SIC includes an extra dimension for each ensemble member.
        if len(sic.shape) == 3:
            valid_axes = tuple(i for i in range(len(sic.shape)))
        elif len(sic.shape) == 4:
            valid_axes = tuple(i for i in range(len(sic.shape)) if i != ensemble_axis)

        # Divide by 10^6 to get:
        # (units in 10^6 km^2)
        # Sum across all axes except ensemble
        sea_ice_extent = sic.sum(axis=valid_axes) / 1E6

        return sea_ice_extent

    def compute_daily_sea_ice_extent(self, method="mean", grid_cell_area=25*25, threshold=0.15, plot=False):
        """Daily Sea Ice Extent.

        Computes the total extent (SIC>15%) for each day.
        """
        if method == "mean":
            sic = self.xarr.sic_mean
        elif method == "ensemble":
            sic = self.xarr.sic

        kwargs = {"grid_cell_area": grid_cell_area, "threshold": threshold, "plot": plot}
        sea_ice_extent_daily = np.asarray([sic.isel(leadtime=day-1).map_blocks(self.compute_sea_ice_extent, kwargs=kwargs).values for day in self.xarr.leadtime]);
        forecast_dates = pd.to_datetime(self.xarr.forecast_date)

        if method == "mean":
            sea_ice_extent_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_extent_daily_mean = (["day"], sea_ice_extent_daily),
                ),
                coords=dict(
                    day=forecast_dates,
                )
            )
        elif method == "ensemble":
            sea_ice_extent_daily_mean = sea_ice_extent_daily.mean(axis=1)
            sea_ice_extent_daily_stddev = sea_ice_extent_daily.std(axis=1)

            sea_ice_extent_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_extent_daily = (["day", "ensemble"], sea_ice_extent_daily),
                    sea_ice_extent_daily_mean = (["day"], sea_ice_extent_daily_mean),
                    sea_ice_extent_daily_stddev = (["day"], sea_ice_extent_daily_stddev),
                ),
                coords=dict(
                    day=forecast_dates,
                    ensemble=list(range(sea_ice_extent_daily.shape[1]))
                )
            )

        sea_ice_extent_daily_ds["sea_ice_extent_daily_mean"].attrs["long_name"] = "Total Sea-Ice Extent for each day"
        sea_ice_extent_daily_ds["sea_ice_extent_daily_mean"].attrs["units"] = "10⁶ km²"

        return sea_ice_extent_daily_ds

    def compute_monthly_sea_ice_extent(self, method="mean", grid_cell_area=25*25, threshold=0.15, plot=False):
        """Monthly Sea Ice Extent from daily.

        Computes the total extent (SIC>15%) for each day, then, averages the extent
        from each of the days of the month into a monthly average extent.

        To be consistent with NSIDC Sea Ice Index extent.
        """
        sea_ice_extent_daily_ds = self.compute_daily_sea_ice_extent(method=method, grid_cell_area=grid_cell_area, threshold=threshold, plot=plot)

        sea_ice_extent_monthly_ds = sea_ice_extent_daily_ds.sea_ice_extent_daily_mean.groupby(sea_ice_extent_daily_ds.day.dt.month).mean().rename("sea_ice_extent_monthly_mean")

        self.xarr["sea_ice_extent_monthly_mean"] = sea_ice_extent_monthly_ds
        self.xarr["sea_ice_extent_monthly_mean"].attrs["long_name"] = "Total Sea-Ice Extent for each month, averaged from daily"
        self.xarr["sea_ice_extent_monthly_mean"].attrs["units"] = "10⁶ km²"
        self.xarr.month.attrs = {"long_name": "months for which mean sea ice extent are computed for"}

        return sea_ice_extent_monthly_ds
