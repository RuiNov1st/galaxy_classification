from search_tool import hips_fits_url_catalog,save_fits
import pandas as pd
import asyncio
from astroquery.simbad import Simbad

# test for catalog input
async def catalog_test():
    df  = pd.read_csv('../dataset/galaxy_catalog/Skyserver_SQL5_30_2024 9_37_50 AM.csv')
    res = df[6:10]
    url_list,filename_list = hips_fits_url_catalog(res)
    for i in range(len(url_list)):
        # get image
        status_code = await save_fits(url_list[i],filename_list[i])

asyncio.run(catalog_test())
