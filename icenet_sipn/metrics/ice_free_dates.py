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
from ..utils import drop_variables

class IceFreeDates(ABC):
    """To compute/plot IFD from individual ensemble member outputs
    """
    def __init__(self) -> None:
        pass

    def clear_ifd(self):
        """Drop `ice free dates` related variables if previously saved in dataset.
        """
        variable_names = [
            "ice_free_dates",
            "ice_free_dates_mean",
            "ice_free_dates_stddev",
        ]
        self.xarr = drop_variables(self.xarr, variable_names)

    @staticmethod
    def get_cmap(cmap="viridis"):
        """Create a colourmap including out-of-range values"""
        cmap = plt.get_cmap(cmap)
        cmap.set_under("grey")
        colours = cmap(np.linspace(0, 1, 257))
        new_colours = np.vstack((
            [1, 1, 1, 0],       # White for under colour-range
            colours,            # Normal colourmap
            [0.5, 0.5, 0.5, 1]  # Dark grey for over colour-range
            ))
        new_cmap = mcolors.ListedColormap(new_colours)
        return new_cmap

    def plot_ice_free_dates_from_sic_mean(self, ifd_data, threshold=0.15, figsize=(8, 8)):
        pole = self.get_pole
        ifd = ifd_data - self.date.timetuple().tm_yday

        # Identify first day of each month
        dates = self.xarr.forecast_date.values.astype("datetime64[D]")

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        new_cmap = self.get_cmap()

        # Define new boundary for outside of vmax
        vmin=self.xarr.leadtime[0]
        vmax=self.xarr.leadtime[-1]
        boundaries = np.linspace(vmin, vmax, 257)
        norm = mcolors.BoundaryNorm(boundaries, new_cmap.N)

        source_crs = ccrs.LambertAzimuthalEqualArea(central_latitude=pole*90, central_longitude=0)
        target_crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": source_crs},
                               layout="tight",
                               )

        img2 = ifd.plot.pcolormesh("lon", "lat",
                                   vmin=vmin, vmax=vmax,
                                   ax=ax,
                                   cmap=new_cmap,
                                   transform=target_crs,
                                   norm=norm,
                                   alpha=1.0,
                                   add_labels=False,
                                   )

        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="lightgrey")

        # Customise colourbar
        cbar = img2.colorbar
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        ax.set_title(f"Ice-Free Dates (IFD{int(threshold*100)})")

        plt.show()


    def plot_ice_free_dates_from_sic_ensemble(self, ifd_data, dates, index):
        """Interactive plot of Ice-Free Dates 15% for all ensemble members.
        """
        # Identify first day of each month
        dates = dates.astype("datetime64[D]")

        start_date = dt.datetime.strptime(str(dates[0]), "%Y-%m-%d")
        ifd = ifd_data - start_date.timetuple().tm_yday

        land_mask = Masks(south=False, north=True).get_land_mask()
        mask = xr.DataArray(~land_mask, coords=[ifd.coords["yc"], ifd.coords["xc"]])

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        new_cmap = self.get_cmap()

        leadtimes = [day for day in range(len(dates))]
        vmin=leadtimes[0]
        vmax=leadtimes[-1]
        boundaries = np.linspace(vmin, vmax, 257)
        norm = mcolors.BoundaryNorm(boundaries, new_cmap.N)

        img1 = mask.plot.imshow(levels=[0, 1], colors="Grey", alpha=0.2, add_colorbar=False)
        img2 = ifd[index].plot.imshow(vmin=vmin, vmax=vmax, cmap=new_cmap, norm=norm, alpha=1.0,
                                    add_labels=False)

        # Customise colourbar
        cbar = img2.colorbar
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        title = f"Ice-Free Dates (IFD15) | Ensemble {index}"

        plt.title(title)
        plt.show()

    def plot_ice_free_dates_from_ensemble_mean_stddev(self, ifd_mean, ifd_stddev, dates):
        """Ice-Free Dates 15% for an ensemble mean and corresponding standard deviation.
        """
        # Identify first day of each month
        dates = dates.astype("datetime64[D]")

        start_date = dt.datetime.strptime(str(dates[0]), "%Y-%m-%d")
        ifd = ifd_mean - start_date.timetuple().tm_yday

        land_mask = Masks(south=False, north=True).get_land_mask()
        mask = xr.DataArray(~land_mask, coords=[ifd.coords["yc"], ifd.coords["xc"]])

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        new_cmap = self.get_cmap()

        leadtimes = [day for day in range(len(dates))]
        vmin=leadtimes[0]
        vmax=leadtimes[-1]
        boundaries = np.linspace(vmin, vmax, 257)
        norm = mcolors.BoundaryNorm(boundaries, new_cmap.N)

        img1 = mask.plot.imshow(levels=[0, 1], colors="Grey", alpha=0.2, add_colorbar=False)
        img2 = ifd.plot.imshow(vmin=vmin, vmax=vmax, cmap=new_cmap, norm=norm, alpha=1.0,
                                    add_labels=False)

        # Customise colourbar
        cbar = img2.colorbar
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        plt.title(f"Ice-Free Dates (IFD15) Mean")

        plt.show()

        img1 = mask.plot.imshow(levels=[0, 1], colors="Grey", alpha=0.2, add_colorbar=False)
        img2 = ifd_stddev.plot.imshow(vmin=vmin, vmax=vmax, cmap=new_cmap, norm=norm, alpha=1.0,
                                    add_labels=False)

        # Customise colourbar
        cbar = img2.colorbar
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.set_label("Days")

        plt.title(f"Ice-Free Dates (IFD15) Standard Deviation")

        plt.show()

    def compute_ice_free_dates_for_single_sic(self, sic, threshold=0.15):
        """Ice Free Dates for a single ensemble member or sic_mean.

        The first day the SIC drops below `threshold` from mean SIC.

        Args:
            threshold (optional): Threshold of Sea Ice Concentration (ranges from 0 to 1) to compute IFD for.
                Defaults to 0.15.

        Returns:
            ice_free_dates: xarray.DataArray field with the mean of ice_free_dates_ensemble.
                Dimensions: [yc, xc]

        Raises:
            AssertionError: If `threshold` is not between 0 and 1 (exclusive).
        """
        assert 0 < threshold < 1, f"threshold={threshold}, threshold must be between 0 and 1"

        land_mask = Masks(south=self.south, north=self.north).get_land_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[land_mask] = np.nan
        land_mask_nan[~land_mask] = 1.0

        ice_free_dates = ( sic <= threshold ).argmax(dim="leadtime")*land_mask_nan#*start_mask.data#*end_mask.data

        threshold_never_met = (( sic <= threshold ).sum(dim="leadtime") == 0)

        # Push areas where the threshold is never met to above upper range of leadtime
        # Doing this so that plot will show the colour set in colourbar's upper.
        ice_free_dates = ice_free_dates.where(~threshold_never_met, other=100)*land_mask_nan#*start_mask.data#*end_mask.data

        # Add number of days since start of the year
        return ice_free_dates + self.date.timetuple().tm_yday


    def compute_ice_free_dates_from_sic_mean(self, threshold=0.15, plot=True):
        """Ice Free Dates computed from `sic_mean`.

        The first day the SIC drops below `threshold` from mean SIC.

        Args:
            threshold (optional): Threshold of Sea Ice Concentration (ranges from 0 to 1) to compute IFD for.
                Defaults to 0.15.
            plot (optional): Whether to generate a plot of the result.
                Defaults to True.

        Returns:
            ice_free_dates: xarray.DataArray field with the mean of ice_free_dates_ensemble.
                Dimensions: [yc, xc, leadtime]

        Raises:
            AssertionError: If `threshold` is not between 0 and 1 (exclusive).
        """
        assert 0 < threshold < 1, f"threshold={threshold}, threshold must be between 0 and 1"

        sic = self.xarr.sic_mean.isel(time=0)
        ice_free_dates = self.compute_ice_free_dates_for_single_sic(sic, threshold=threshold)

        self.xarr["ice_free_dates_15"] = (("yc", "xc"), ice_free_dates.data)
        self.xarr["ice_free_dates_15"].attrs["long_name"] = "Ice-Free Dates with 15% threshold from SIC mean"

        if plot:
            self.plot_ice_free_dates_from_sic_mean(ice_free_dates, threshold=threshold)

        return ice_free_dates

    def compute_ice_free_dates_from_sic_ensemble(self, threshold=0.15, plot=True):
        """Ice Free Dates across each ensemble members, then take mean across the result.

        The first day the SIC drops below `threshold` across all ensemble predictions.

        Args:
            threshold (optional): Threshold of Sea Ice Concentration (ranges from 0 to 1) to compute IFD for.
                Defaults to 0.15.
            plot (optional): Whether to generate a plot of the result.
                Defaults to True.

        Returns:
            ice_free_dates_ensemble: xarray.DataArray field with Ice Free Dates for all ensemble members.
                Dimensions: [ensemble, yc, xc, leadtime]
            ice_free_dates_mean: xarray.DataArray field with the mean of ice_free_dates_ensemble.
                Dimensions: [yc, xc, leadtime]
            ice_free_dates_stddev: xarray.DataArray field with the std dev of ice_free_dates_ensemble.
                Dimensions: [yc, xc, leadtime]

        Raises:
            AssertionError: If `threshold` is not between 0 and 1 (exclusive).
        """
        assert 0 < threshold < 1, f"threshold={threshold}, threshold must be between 0 and 1"

        xarr = self.xarr
        ensembles = list(range(xarr.ensemble_members.data))

        kwargs = {"threshold": threshold, "plot": False, "dates": xarr.forecast_date.values}
        ice_free_dates_ensemble = np.asarray([xarr.sic.isel(time=0, ensemble=ensemble).map_blocks(self.compute_ice_free_dates_for_single_sic, kwargs=kwargs).values for ensemble in ensembles])
        ice_free_dates_mean = ice_free_dates_ensemble.mean(axis=0)
        ice_free_dates_stddev = ice_free_dates_ensemble.std(axis=0)

        xarr["ice_free_dates_15"] = (("ensemble", "yc", "xc"), ice_free_dates_ensemble.data)
        xarr["ice_free_dates_15"].attrs["long_name"] = "Ice-Free Dates with 15% (IFD15) threshold across each ensemble"
        xarr["ice_free_dates_15_mean"] = (("yc", "xc"), ice_free_dates_mean.data)
        xarr["ice_free_dates_15_mean"].attrs["long_name"] = "Mean from ensemble IFD15 calculation"
        xarr["ice_free_dates_15_stddev"] = (("yc", "xc"), ice_free_dates_stddev.data)
        xarr["ice_free_dates_15_stddev"].attrs["long_name"] = "Standard Deviation from ensemble IFD15 calculation"

        if plot:
            # Plot IFD15 ensemble results
            partial_plot_ice_free_dates = partial(self.plot_ice_free_dates_from_sic_ensemble, xarr["ice_free_dates"], xarr.forecast_date.values)

            def wrapped_plot_func(index):
                """Wrap partial plot func so that it has a __name__ for ipywidgets to work
                with partial func call.
                """
                partial_plot_ice_free_dates(index)

            interact(wrapped_plot_func, index=SelectionSlider(options=ensembles, value=ensembles[0], description="Ensemble"))

            # Plot IFD15 mean and standard deviation results
            self.plot_ice_free_dates_from_ensemble_mean_stddev(xarr["ice_free_dates_mean"], xarr["ice_free_dates_stddev"], xarr.forecast_date.values)

        return ice_free_dates_ensemble, ice_free_dates_mean, ice_free_dates_stddev

    def compute_ice_free_dates_from_sic(self, method: str="mean", threshold: float=0.15, plot: bool=True) -> object:
        """Ice Free Dates (IFD) from Sea Ice Concentration (SIC) field.

        The first day the SIC drops below a certain `threshold`.

        Args:
            method (optional): Approach to take in calculating metric. It must be `mean` or `ensemble`.
                Defaults to `mean`.
            threshold (optional): Threshold of Sea Ice Concentration (ranges from 0 to 1) to compute IFD for.
                Defaults to 0.15.
            plot (optional): Whether to generate a plot of the result.
                Defaults to True.

        Returns:
            ice_free_dates: xarray.DataArray field with Ice Free Dates.

        Raises:
            AssertionError: If `threshold` is not between 0 and 1 (exclusive).
            ValueError: If `method` is not `mean` or `ensemble`
        """

        assert 0 < threshold < 1, f"threshold={threshold}, threshold must be between 0 and 1"

        self.clear_ifd()

        if method == "mean":
            ice_free_dates = self.compute_ice_free_dates_from_sic_mean(plot=plot, threshold=threshold)
        elif method == "ensemble":
            ice_free_dates = self.compute_ice_free_dates_from_sic_ensemble(plot=plot, threshold=threshold)
        else:
            raise ValueError("Expected `method` to be `mean` or `ensemble`")

        return ice_free_dates
