from astropy.io import fits
from PIL import Image
import numpy as np

from urllib.parse import urlencode
from astropy.io import fits
from astropy.utils.data import download_file,clear_download_cache
import pandas as pd
from astroquery.simbad import Simbad
import os
import concurrent.futures
import pyvo
import asyncio

# define your images dir:
image_dir = './images/'

def parse_survey(survey_id:str):
    """
    Parse Survey name to match Aladin Resources: https://aladin.cds.unistra.fr/hips/list.
    If cannot find correct survey name, use DESI-Legacy-Surveys/DR10/color(training set) instead.
    params: 
    - survey_id: survey_id from Aladin imagelayer.creatorDid
    return:
    - survey name in hips list.
    """
    prefix = 'ivo://'
    if survey_id.startswith(prefix):
        return survey_id[len(prefix):]
    else:
        print("Can not find {survey_id} data, use DESI-Legacy-Surveys/DR10/color (Default) instead.")
        return 'CDS/P/DESI-Legacy-Surveys/DR10/color'

def hips_fits_url(ra:float,dec:float,fov:float,size:int,survey='CDS/P/DESI-Legacy-Surveys/DR10/color'):
    """
    Use parameters of Aladin's view to generate hips-image-services URL.
    params:
    - ra: deg
    - dec: deg
    - fov: deg
    - size: pixel
    - survey: https://aladin.cds.unistra.fr/hips/list
    return:
    - url: hips-image-services URL
    - filename: combine all information above to be an unique name.
    """
    query_params = { 
        'hips': survey, 
        'ra': ra, 
        'dec': dec, 
        'fov': fov, 
        'width': size, 
        'height': size 
    }                                                                                               
    url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{urlencode(query_params)}' 
    filename=f'{urlencode(query_params)}'
    return url,filename


def hips_fits_url_catalog(catalog_df:pd.DataFrame):
    """
    Load catalog information to generate hips-image-services URL for image labelling.
    params:
    - catalog_df: catalog should contain information as below: ra,dec,fov(optional),size(optional),survey(optional, names should match https://aladin.cds.unistra.fr/hips/list)
    return:
    - url_list: URLs for all coords in catalog.
    - filename_list: filenames for all coords in catalog.
    """
    # coords:
    RA = np.array(catalog_df['ra'])
    DEC = np.array(catalog_df['dec'])
    dataset_size = catalog_df.shape[0]

    # optional data:
    optional_keys = ['fov','size','survey']
    # init
    optional_dict = {
        'fov':[0.01]*dataset_size, # deg
        'size':[256]*dataset_size, # pixel
        'survey':['P/DESI-Legacy-Surveys/DR10/color']*dataset_size # default survey
    }
    # if catalog contain optional information, use it.
    for k in optional_keys:
        if k in list(catalog_df.keys()):
            optional_dict[k] = catalog_df[k]

    # get download url & filename:
    url_list,filename_list = [],[]
    for t in range(dataset_size):
        url,filename = hips_fits_url(ra=RA[t],dec=DEC[t],fov=optional_dict['fov'][t],size=optional_dict['size'][t],survey=optional_dict['survey'][t])
        url_list.append(url)
        filename_list.append(filename)
    return url_list,filename_list


async def save_fits(url,filename):
    """
    Use URL to download images.
    """
    try:
        # download image:
        image_file = download_file(url,timeout=120,cache=True)
        hdul = fits.open(image_file)
        data = hdul[0].data

        # save image:
        img = data.transpose(1,2,0) # (width,height,3) 
        image = Image.fromarray(img.astype(np.uint8))
        
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        image.save(image_dir+f'{filename}.png')
        print("Saving Image Successfully!")

        # delete cache (Optional):
        # clear_download_cache(url)

        return 200
    except Exception as e:
        print(e)
        return 500


async def query_information(ra,dec,radius=0.05):
    """
    query simbad information with ra & dec coords with 0.05 deg radius.
    params:
    - ra: deg
    - dec: deg
    - radius: deg, default 0.05 deg
    return:
    - Table: information about ra, dec, main_id(name), rvz_redshift(redshift), otype(galaxy), morph_type(Galaxy Morphology), galdim_majaxis(major axis angular size)
    """
    def query_sync(query_adql):
        """
        To match pyvo sync
        param:
        - query_adql: ADQL
        return:
        -result: TAPresult
        """
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")
        result = service.search(query_adql)
        return result
    
    query_adql = f"""SELECT TOP 5
            ra, dec, main_id, rvz_redshift, otype,morph_type,galdim_majaxis
            FROM basic
            WHERE otype = 'Galaxy..'
            AND CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS',{ra},{dec}, {radius})) = 1
            """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        table = await loop.run_in_executor(pool, query_sync, query_adql)
    # table = await Simbad.query_tap(query_adql)
    if len(table) == 0:
        print(f"Can not query associated information for {ra} {dec} within {radius} degree.")
        return []
    else:
        # to make it Serializable.
        table = table.to_table().to_pandas()
        table = table.replace({float('nan'): None, float('inf'): None, float('-inf'): None})
        return table.to_dict(orient='records')











