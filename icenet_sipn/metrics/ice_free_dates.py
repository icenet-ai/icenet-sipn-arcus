import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import xarray as xr

from functools import partial

from icenet.data.sic.mask import Masks
from ipywidgets import interact, IntSlider, SelectionSlider
from ..utils import drop_variables

class IceFreeDates:
    """To compute/plot IFD from individual ensemble member outputs
    """
    def __init__(self) -> None:
        pass

    def clear_vars(self):
        """Drop `ice free dates` related variables if previously saved in dataset.
        """
        variable_names = [
            "ice_free_dates",
            "ice_free_dates_mean",
            "ice_free_dates_stddev",
        ]
        self.xarr = drop_variables(self.xarr, variable_names)

    def plot_ice_free_dates_from_sic_mean(self, ifd_data):
        ifd = ifd_data - self.date.timetuple().tm_yday

        land_mask = Masks(south=False, north=True).get_land_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[~land_mask] = np.nan
        land_mask_nan[land_mask] = 1.0
        land_mask_nan = land_mask_nan.astype(bool)

        mask = xr.DataArray(~land_mask, coords=[ifd.coords["yc"], ifd_data.coords["xc"]])

        # ifd = ifd.where(mask == 0, other=100)

        # Identify first day of each month
        dates = self.xarr.forecast_date.values.astype("datetime64[D]")

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        # Create a colourmap including out-of-range values
        cmap = plt.get_cmap("viridis")
        cmap.set_under("grey")
        colours = cmap(np.linspace(0, 1, 258))
        new_colours = np.vstack(([1, 1, 1, 0], [1, 1, 1, 0], colours, [0.5, 0.5, 0.5, 1]))
        new_cmap = mcolors.ListedColormap(new_colours)

        # Define new boundary for outside of vmax
        vmin=self.xarr.leadtime[0]
        vmax=self.xarr.leadtime[-1]
        boundaries = np.linspace(vmin, vmax, 259)
        norm = mcolors.BoundaryNorm(boundaries, new_cmap.N)


        fig, ax = plt.subplots()

        img1 = mask.plot.imshow(levels=[0, 1], colors="Grey", alpha=0.2, add_colorbar=False)
        # img1 = plt.imshow(land_mask_nan)
        img2 = ifd.plot.imshow(vmin=vmin, vmax=vmax, cmap=new_cmap, norm=norm, alpha=1.0,
                                    add_labels=False)


        # Customise colourbar
        cbar = img2.colorbar
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        # Hide x and y-axis labels
        img2.axes.xaxis.set_visible(False)
        img2.axes.yaxis.set_visible(False)
        # ax.axis("off")

        fig.suptitle("Ice-Free Dates (IFD15)")

        plt.show()


















        # land_mask = Masks(south=False, north=True).get_land_mask()
        # land_mask_nan = land_mask.astype(float)
        # land_mask_nan[land_mask] = np.nan
        # land_mask_nan[~land_mask] = 1.0
        # # land_mask_nan = land_mask_nan.astype(bool)

        # mask = xr.DataArray(land_mask, coords=[ifd_data.coords["yc"], ifd_data.coords["xc"]])

        # # ifd_data = ifd_data.where(mask == 0, other=100)

        # # Identify first day of each month
        # dates = self.xarr.forecast_date.values.astype("datetime64[D]")

        # # Get array of unique months
        # unique_months = np.unique(dates.astype("datetime64[M]"))

        # # Include first day of the month
        # first_of_months = unique_months + np.timedelta64(0, "D")

        # # Create a colourmap including out-of-range values
        # cmap = plt.get_cmap("viridis")
        # cmap.set_under("grey")
        # colours = cmap(np.linspace(0, 1, 258))
        # new_colours = np.vstack(([1, 1, 1, 1], [1, 1, 1, 1], colours, [0.5, 0.5, 0.5, 1]))
        # new_cmap = mcolors.ListedColormap(new_colours)

        # # Define new boundary for outside of vmax
        # vmin=self.xarr.leadtime[0]
        # vmax=self.xarr.leadtime[-1]
        # boundaries = np.linspace(vmin, vmax, 259)
        # norm = mcolors.BoundaryNorm(boundaries, new_cmap.N)

        # if type(ifd_data) == np.ndarray:
        #     img = plt.imshow(ifd_data)
        # else:
        #     img = ifd_data.plot.imshow(vmin=vmin, vmax=vmax, cmap=new_cmap, norm=norm)

        # # Customise colourbar
        # cbar = img.colorbar
        # cbar.set_label("")
        # tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        # cbar.set_ticks(tick_positions)
        # cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        # # Hide x and y-axis labels
        # img.axes.xaxis.set_visible(False)
        # img.axes.yaxis.set_visible(False)

        # plt.title("Ice-Free Dates (IFD15)")

    @staticmethod
    def plot_ice_free_dates_from_sic_ensemble(ifd_data, dates, index):
        """Interactive plot of Ice-Free Dates 15% for all ensemble members.
        """
        # Identify first day of each month
        dates = dates.astype("datetime64[D]")

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        img = ifd_data[index].plot.imshow()

        # Customise colourbar
        cbar = img.colorbar
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        # Hide x and y-axis labels
        img.axes.xaxis.set_visible(False)
        img.axes.yaxis.set_visible(False)

        title = f"Ice-Free Dates (IFD15) | Ensemble {index}"

        plt.title(title)
        plt.show()

    @staticmethod
    def plot_ice_free_dates_from_ensemble_mean_stddev(ifd_mean, ifd_stddev, dates):
        """Ice-Free Dates 15% for an ensemble mean and corresponding standard deviation.
        """
        # Identify first day of each month
        dates = dates.astype("datetime64[D]")

        # Get array of unique months
        unique_months = np.unique(dates.astype("datetime64[M]"))

        # Include first day of the month
        first_of_months = unique_months + np.timedelta64(0, "D")

        img = ifd_mean.plot.imshow()

        # Customise colourbar
        cbar = img.colorbar
        cbar.set_label("")
        tick_positions = [(date - dates[0]).astype(int) for date in first_of_months]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([date.astype(dt.datetime).strftime("%b %-d") for date in first_of_months])

        plt.title(f"Ice-Free Dates (IFD15) Mean")

        # Hide x and y-axis labels
        img.axes.xaxis.set_visible(False)
        img.axes.yaxis.set_visible(False)

        plt.show()

        img = ifd_stddev.plot.imshow()

        # Customise colourbar
        cbar = img.colorbar
        cbar.set_label("Days")

        img.axes.xaxis.set_visible(False)
        img.axes.yaxis.set_visible(False)

        plt.title(f"Ice-Free Dates (IFD15) Standard Deviation")

        plt.show()

    def compute_ice_free_dates_for_single_sic(self, sic, dates=None, threshold=0.15, plot=False):
        """Ice Free Dates for a single ensemble member or sic_mean.

        The first day the SIC drops below 15% (IFD15).
        """
        land_mask = Masks(south=False, north=True).get_land_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[land_mask] = np.nan
        land_mask_nan[~land_mask] = 1.0

        print("Shape:", sic.shape)
        # Mask region where ocean at start
        start_mask = sic.isel(leadtime=0).where(sic.isel(leadtime=0)>threshold, np.nan)
        end_mask = sic.isel(leadtime=-1).where(sic.isel(leadtime=-1)<threshold, np.nan)
        # print("Start mask shape:", start_mask.shape)
        # Find the index of the first day where SIC hits the threshold
        ice_free_dates = ( sic <= threshold ).argmax(dim="leadtime")*land_mask_nan#*start_mask.data#*end_mask.data

        threshold_never_met = (( sic <= threshold ).sum(dim="leadtime") == 0)

        ice_free_dates = ice_free_dates.where(~threshold_never_met, other=100)*land_mask_nan#*start_mask.data#*end_mask.data

        # dates = self.xarr.forecast_date.values.astype("datetime64[D]").astype(dt.datetime)
        # dates = [dt.strptime(date, '%Y-%m-%d') for date in dates]

        # ice_free_dates = np.full(sic[..., 0].shape, -1)
        # print("IFD shape:", ice_free_dates.shape)



        # for i in range(ice_free_dates.shape[0]):
        #     for j in range(ice_free_dates.shape[1]):
        #         for k in range(sic.shape[2]):
        #             image = sic[..., k]
        #             if image[i, j] < threshold:
        #                 ice_free_dates[i, j] = dates[k].timetuple().tm_yday
        #                 break




        # first_occurence = np.argmax(sic.data < threshold, axis=2)
        # day_of_year = np.array([date.timetuple().tm_yday for date in dates])
        # ice_free_dates[:] = day_of_year[first_occurence]

        return ice_free_dates + self.date.timetuple().tm_yday

    def compute_ice_free_dates_from_sic_mean(self, threshold=0.15, plot=True):
        """Ice Free Dates for `sic_mean`.

        The first day the SIC drops below 15% (IFD15).
        """
        sic = self.xarr.sic_mean.isel(time=0)
        ice_free_dates = self.compute_ice_free_dates_for_single_sic(sic)

        self.xarr["ice_free_dates"] = (("yc", "xc"), ice_free_dates.data)
        self.xarr["ice_free_dates"].attrs["long_name"] = "Ice-Free Dates with 15% threshold from SIC mean"

        if plot:
            self.plot_ice_free_dates_from_sic_mean(ice_free_dates)

        return ice_free_dates

    def compute_ice_free_dates_from_sic_ensemble(self, threshold=0.15, plot=True):
        """Ice Free Dates across each ensemble members, then take mean.

        The first day the SIC drops below 15% (IFD15).
        """
        xarr = self.xarr
        ensembles = list(range(xarr.ensemble_members.data))

        kwargs = {"threshold": threshold, "plot": False, "dates": xarr.forecast_date.values}
        ice_free_dates_ensemble = np.asarray([xarr.sic.isel(time=0, ensemble=ensemble).map_blocks(self.compute_ice_free_dates_for_single_sic, kwargs=kwargs).values for ensemble in ensembles])
        ice_free_dates_mean = ice_free_dates_ensemble.mean(axis=0)
        ice_free_dates_stddev = ice_free_dates_ensemble.std(axis=0)

        xarr["ice_free_dates"] = (("ensemble", "yc", "xc"), ice_free_dates_ensemble.data)
        xarr["ice_free_dates"].attrs["long_name"] = "Ice-Free Dates with 15% (IFD15) threshold across each ensemble"
        xarr["ice_free_dates_mean"] = (("yc", "xc"), ice_free_dates_mean.data)
        xarr["ice_free_dates_mean"].attrs["long_name"] = "Mean from ensemble IFD15 calculation"
        xarr["ice_free_dates_stddev"] = (("yc", "xc"), ice_free_dates_stddev.data)
        xarr["ice_free_dates_stddev"].attrs["long_name"] = "Standard Deviation from ensemble IFD15 calculation"

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

    def compute_ice_free_dates_from_sic(self, method="mean", threshold=0.15, plot=True):
        """Ice Free Dates.

        The first day the SIC drops below 15% (IFD15).
        """

        self.clear_vars()

        if method == "mean":
            print("Method mean")
            ice_free_dates = self.compute_ice_free_dates_from_sic_mean(plot=plot)
        elif method == "ensemble":
            print("Method ensemble")
            ice_free_dates = self.compute_ice_free_dates_from_sic_ensemble(plot=plot)

        return ice_free_dates
