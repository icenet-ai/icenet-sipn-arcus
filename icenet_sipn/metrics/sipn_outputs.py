import datetime as dt

from ..process.icenet import IceNetOutputPostProcess
from .sea_ice_extent import SeaIceExtent
from .ice_free_dates import IceFreeDates


class SIPNOutputs(IceNetOutputPostProcess,
                  SeaIceExtent,
                  IceFreeDates,
                  ):
    """SIPN ARCUS Sea Ice Outlook Submission for IceNet (with daily averaging 
    up to 93 day leadtime)
    
    Refer here: https://www.arcus.org/sipn/sea-ice-outlook/2023/june
    """
    def __init__(self, hemisphere: str, prediction_path: str, date: dt.date) -> None:
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

        super().__init__(prediction_path, date)
        self.create_ensemble_dataset()


    def check_forecast_period(self):
        """Used to check that the forecast period is within June-November.
        """
        raise NotImplementedError

    def get_data(self):
        return self.xarr
