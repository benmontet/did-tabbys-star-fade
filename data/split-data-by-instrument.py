import numpy as np
from astropy.table import Table, join
from astropy.time import Time


def jd2year(jd):
    return int(np.floor(Time(jd, format="jd").decimalyear))


if __name__ == "__main__":
    series_tbl = Table.read("dasch-plate-series.csv", format="ascii.csv")

    tbl = Table.read("20:06:15.457_+44:27:24.61_APASS_J200615.5+442725_0001.xml.gz")
    series_ids = np.unique(tbl["seriesId"])

    summary = []
    for seriesid in series_ids:
        mask = tbl["seriesId"] == seriesid
        tbl[mask].write("series/series-{}.csv".format(seriesid), format="ascii.csv")
        summary.append({
                                'seriesId': seriesid,
                                'year_begin': jd2year(tbl[mask]["ExposureDate"].min()),
                                'year_end': jd2year(tbl[mask]["ExposureDate"].max()),
                                'datapoints' : mask.sum()
                        })

    summary_tbl = Table(summary)
    summary_tbl = join(summary_tbl, series_tbl,
                       keys="seriesId", join_type="left")
    summary_tbl.sort("datapoints")
    summary_tbl.reverse()
    summary_tbl["series", "seriesId", "datapoints", "year_begin", "year_end", "aperture", "telescope"].write("dasch-data-summary.csv", format="ascii.csv")
