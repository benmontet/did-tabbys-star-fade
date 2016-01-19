"""This script prepares a pandas dataframe containing the DASCH data.

It also creates a summary csv file with details on each series.
"""
import numpy as np
import pandas as pd

from astropy.table import Table, join
from astropy.time import Time


def jd2year(jd):
    """Returns the year corresponding to a julian date."""
    return int(np.floor(Time(jd, format="jd").decimalyear))


if __name__ == "__main__":
    series_tbl = Table.read("dasch-plate-series.csv", format="ascii.csv")

    tbl = Table.read("20:06:15.457_+44:27:24.61_APASS_J200615.5+442725_0001.xml.gz")
    # Add useful columns
    tbl['isodate'] = Time(tbl['ExposureDate'], format="jd").iso
    tbl['decimalyear'] = Time(tbl['ExposureDate'], format="jd").decimalyear

    series_ids = np.unique(tbl["seriesId"])

    summary = []
    dataframes = []
    for seriesid in series_ids:
        # AFLAGS are described here:
        # http://dasch.rc.fas.harvard.edu/database.php#AFLAGS_ext
        # Schaefer rejects data points with AFLAGS > 9000
        mask = (
                (tbl["seriesId"] == seriesid) &
                (tbl["AFLAGS"] <= 9000) &
                (tbl["magcal_local_rms"] <= 0.33) &
                ((tbl["magcal_magdep"] + 0.2) < tbl["limiting_mag_local"])
                )

        if mask.sum() > 1:
            dataframes.append(tbl[mask].to_pandas())
            tbl[mask].write("series/series-{}.csv".format(seriesid), format="ascii.csv")
            summary.append({
                                    'seriesId': seriesid,
                                    'year_begin': jd2year(tbl[mask]["ExposureDate"].min()),
                                    'year_end': jd2year(tbl[mask]["ExposureDate"].max()),
                                    'good_datapoints' : mask.sum()
                            })
    df = pd.concat(dataframes)
    output_fn = "preprocessed-data.csv"
    print("Writing {}".format(output_fn))
    df.to_csv(output_fn)

    summary_tbl = Table(summary)
    summary_tbl = join(summary_tbl, series_tbl,
                       keys="seriesId", join_type="left")
    summary_tbl.sort("good_datapoints")
    summary_tbl.reverse()
    summary_tbl["series", "seriesId", "good_datapoints", "year_begin", "year_end", "aperture", "telescope"].write("dasch-data-summary.csv", format="ascii.csv")
