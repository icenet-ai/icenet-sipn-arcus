import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from abc import ABC
from functools import partial

from icenet.data.sic.mask import Masks
from ipywidgets import interact, SelectionSlider
from matplotlib.animation import FuncAnimation

MONTHS = {
    0: "January",
    1: "February",
    2: "March",
    3: "April",
    4: "May",
    5: "June",
    6: "July",
    7: "August",
    8: "September",
    9: "October",
    10: "November",
    11: "December"
    }

def init_plot(pole, data_crs=None, figsize=(8, 8)):
    data_crs = ccrs.LambertAzimuthalEqualArea(central_latitude=pole*90, central_longitude=0) if data_crs is None else data_crs
    fig, ax = plt.subplots(figsize=figsize,
                            subplot_kw={"projection": data_crs},
                            layout="tight",
                            )
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.LAND, zorder=100, facecolor="lightgrey")
    return fig, ax

class SeaIceProbability(ABC):
    """Sea Ice Probability (SIP)

    The probability that Sea Ice Concentration is over a given threshold in each grid cell as part of an ensemble prediction.
    """
    # def __init__(self);
    #     self.fig, self.ax = init_plot(pole=self.get_pole)

    def plot_sea_ice_probability_for_single_leadtime(self, index, sip_data, threshold, aggregate="daily"):
        print(index)
        self.ax.clear()
        if aggregate == "daily":
            sip = sip_data.isel(day=index)
        elif aggregate == "monthly":
            sip = sip_data.sel(month=index)

        cmap = plt.get_cmap("viridis")
        cmap.set_bad('lightgrey')

        target_crs = ccrs.PlateCarree()

        add_colorbar = True if index == 0 else False

        ax = self.ax
        img = sip.plot.pcolormesh("lon", "lat",
                                   ax=ax,
                                   transform=target_crs,
                                   alpha=1.0,
                                   add_labels=False,
                                   cmap=cmap,
                                   add_colorbar=add_colorbar,
                                   )

        ax.set_title(f"{aggregate.capitalize()} Sea-Ice Probability {int(threshold*100)}%")

    def plot_sea_ice_probability(self, threshold=0.15, aggregate="daily"):
        sip = self.xarr["sea_ice_probability"]

        partial_plot_sea_ice_probability = partial(self.plot_sea_ice_probability_for_single_leadtime, sip)

        def wrapped_plot_func(index):
            """Wrap partial plot func so that it has a __name__ for ipywidgets to work
            with partial func call.
            """
            partial_plot_sea_ice_probability(index, threshold=threshold, aggregate=aggregate)

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

        # plt.show()

    def animate_sea_ice_probability(self, threshold=0.15, aggregate="daily"):
        self.fig, self.ax = init_plot(pole=self.get_pole)

        sip = self.xarr["sea_ice_probability"]
        def update(index):
            if aggregate == "daily":
                data = sip.isel(day=index)
                subtitle = self.xarr.day.dt.strftime("%Y/%m/%d").data[index]
            elif aggregate == "monthly":
                data = sip.isel(month=index)
                subtitle = MONTHS[int(data.month.data)]

            image.set_array(data)
            image_title.set_text(f"{aggregate.capitalize()} Sea-Ice Probability {int(threshold*100)}%\n{subtitle}")
            return image, image_title

        if aggregate == "daily":
            frames = len(sip.day)
            subtitle = self.xarr.day.dt.strftime("%Y-%m-%d").data
            sip_start = sip.isel(day=0)
            fps = 10
        elif aggregate == "monthly":
            frames = len(sip.month.data)
            available_months = self.xarr.month
            subtitle = MONTHS[int(available_months.isel(month=0).data)]
            sip_start = sip.isel(month=0)
            fps = 1
        else:
            raise NotImplementedError

        cmap = plt.get_cmap("viridis")
        cmap.set_bad('lightgrey')

        image = sip_start.plot.pcolormesh("lon", "lat",
                                   ax=self.ax,
                                   transform=ccrs.PlateCarree(),
                                   alpha=1.0,
                                   add_labels=False,
                                   cmap=cmap,
                                   )

        image_title = self.ax.set_title(f"{aggregate.capitalize()} Sea-Ice Probability {int(threshold*100)}%\n{subtitle[0]}")

        anim = FuncAnimation(self.fig, update, frames=frames)

        anim.save(f"SIP_{int(threshold*100)}_{aggregate}_animation.mp4", writer="ffmpeg", fps=fps)


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

        # self.plot_sea_ice_probability(threshold=threshold, aggregate=aggregate)
        self.animate_sea_ice_probability(threshold=threshold, aggregate=aggregate)

        return sea_ice_probability
