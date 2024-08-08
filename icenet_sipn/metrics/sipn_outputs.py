import datetime as dt
import pandas as pd

from icenet.plotting.utils import get_obs_da

from ..process.icenet import IceNetOutputPostProcess
from .sea_ice_extent import SeaIceExtent
from .sea_ice_probability import SeaIceProbability
from .ice_free_dates import IceFreeDates


class SIPNOutputs(IceNetOutputPostProcess,
                  SeaIceExtent,
                  SeaIceProbability,
                  IceFreeDates,
                  ):
    """SIPN ARCUS Sea Ice Outlook Submission for IceNet (with daily averaging 
    up to 93 day leadtime)
    
    Refer here: https://www.arcus.org/sipn/sea-ice-outlook/2023/june
    """
    def __init__(self, hemisphere: str, prediction_pipeline_path: str,
                 prediction_name: str, date: dt.date) -> None:
        """
        Args:
            prediction_path: Path to the numpy prediction outputs
        """
        self.north = False
        self.south = False
        self.hemisphere = hemisphere
        if hemisphere.casefold() == "north":
            self.north = True
        elif hemisphere.casefold() == "south":
            self.south = True
        else:
            raise Exception("Incorrect hemisphere specified, should be `north` or `south`")

        super().__init__(prediction_pipeline_path, prediction_name, date)
        self.create_ensemble_dataset()

        # Get Observational data, if it exists.
        self.obs = get_obs_da(
            self.hemisphere,
            pd.to_datetime(date) + dt.timedelta(days=1),
            pd.to_datetime(date) +
            dt.timedelta(days=int(self.xarr.leadtime.values.max())))


    def check_forecast_period(self):
        """Used to check that the forecast period is within June-November.
        """
        raise NotImplementedError

    def get_data(self):
        return self.xarr
