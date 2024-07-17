import datetime as dt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

from abc import ABC
from functools import partial

from matplotlib.ticker import MaxNLocator
from icenet.data.sic.mask import Masks
from ipywidgets import interact, IntSlider, SelectionSlider

class SeaIceProbability(ABC):
    """Sea Ice Probability (SIP)

    The probability that Sea Ice Concentration is over a given threshold in each grid cell as part of an ensemble prediction.
    """

    def plot_sea_ice_probability_for_single_leadtime(self, sip_data, index, aggregate="daily", figsize=(8, 8)):
        pole = self.get_pole

        if aggregate == "daily":
            sip = sip_data.isel(day=index)
        elif aggregate == "monthly":
            sip = sip_data.sel(month=index)

        cmap = plt.get_cmap("viridis")
        cmap.set_bad('lightgrey')

        source_crs = ccrs.LambertAzimuthalEqualArea(central_latitude=pole*90, central_longitude=0)
        target_crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": source_crs},
                               layout="tight",
                               )

        img = sip.plot.pcolormesh("lon", "lat",
                                   ax=ax,
                                   transform=target_crs,
                                   alpha=1.0,
                                   add_labels=False,
                                   cmap=cmap,
                                   )

        ax.coastlines()
        # ax.add_feature(cfeature.LAND, zorder=100, facecolor="lightgrey")

        ax.set_title(f"Sea-Ice Probability")

        plt.show()

    def plot_sea_ice_probability(self, aggregate="daily"):
        sip = self.xarr["sea_ice_probability"]

        partial_plot_sea_ice_probability = partial(self.plot_sea_ice_probability_for_single_leadtime, sip)

        def wrapped_plot_func(index):
            """Wrap partial plot func so that it has a __name__ for ipywidgets to work
            with partial func call.
            """
            partial_plot_sea_ice_probability(index, aggregate=aggregate)

        sip = self.xarr["sea_ice_probability"]
        if aggregate == "daily":
            description = "Day"
            leadtimes = list(range(len(sip.day)))
        elif aggregate == "monthly":
            description = "Month"
            leadtimes = sip.month.data
        else:
            raise NotImplementedError
        interact(wrapped_plot_func, index=SelectionSlider(options=leadtimes, value=leadtimes[0], description=description))


    def compute_sea_ice_probability(self, threshold: float=0.15, aggregate: str="daily", date_range=None) -> xr.DataArray:
        """Sea Ice Probability (SIP) from ensemble predictions.

        Args:
            threshold (optional): Threshold of Sea Ice Concentration (ranges from 0 to 1) to compute IFD for.
                Defaults to 0.15.
            aggregate (optional): Whether to average over days or months.
                Defaults to "daily".
            date_range (optional): Range of dates (from available forecast dates) to compute SIP for.
                Defaults to None (i.e., all available dates).

        Returns:
            sea_ice_probability: Field with SIP from ensemble members.
                Dimensions: [yc, xc, leadtime]
        """
        land_mask = Masks(south=self.south, north=self.north).get_land_mask()

        # if aggregate == "daily":
        #     sic = self.xarr.sic.isel(time=0)
        # elif aggregate == "monthly":
        #     forecast_dates = pd.to_datetime(self.xarr.forecast_date)
        #     sic = sic.groupby(forecast_dates.dt.month).mean()
        # else:
        #     raise NotImplementedError

        forecast_dates = pd.to_datetime(self.xarr.forecast_date)

        sic_ds = xr.Dataset(
            data_vars=dict(
                sic = (["ensemble", "yc", "xc", "day"], self.xarr.sic.isel(time=0).data)
            ),
            coords=dict(
                ensemble=list(range(self.xarr.ensemble_members.data)),
                xc=self.xarr.xc.data,
                yc=self.xarr.yc.data,
                lat=(("yc", "xc"), self.xarr.lat.data),
                lon=(("yc", "xc"), self.xarr.lon.data),
                day=forecast_dates,
            )
        )

        # sic = self.xarr.sic
        # print((sic.leadtime + self.date.timetuple().tm_yday))
        if aggregate == "monthly":
            sic_ds = sic_ds.groupby(sic_ds.day.dt.month).mean()#.rename({"month": "leadtime"})

        ensemble_axis = sic_ds.sic.get_axis_num("ensemble")

        sea_ice_probability = (sic_ds.sic >= threshold).astype(int).mean(axis=ensemble_axis)

        if aggregate == "monthly":
            sea_ice_probability = sea_ice_probability.transpose("yc", "xc", "month")
        elif aggregate == "daily":
            sea_ice_probability = sea_ice_probability.transpose("yc", "xc", "day")
        
        sea_ice_probability = sea_ice_probability.where(land_mask[..., np.newaxis]==0, other=np.nan)

        self.xarr["sea_ice_probability"] = sea_ice_probability
        print(sea_ice_probability.shape)
        self.plot_sea_ice_probability(aggregate=aggregate)



        # # Shift the leading leadtime dimension to the last to match icenet output
        # sic_data = np.moveaxis(sea_ice_probability.data, 0, -1)
        # self.xarr["sea_ice_probability"] = (("yc", "xc", "leadtime"), sic_data)

        # self.plot_sea_ice_probability()

        # return sea_ice_probability


        # sea_ice_probability = (sic >= threshold).astype(int).mean(axis=ensemble_axis)
        # sea_ice_probability = sea_ice_probability.where(land_mask[..., np.newaxis]==0, other=np.nan)

        # self.xarr["sea_ice_probability"] = (("yc", "xc", "leadtime"), sea_ice_probability.data)

        # self.plot_sea_ice_probability()

        # return sea_ice_probability
