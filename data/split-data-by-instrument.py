import numpy as np
from astropy.table import Table


if __name__ == "__main__":
    tbl = Table.read("20:06:15.457_+44:27:24.61_APASS_J200615.5+442725_0001.xml.gz")
    series_ids = np.unique(tbl["seriesId"])
