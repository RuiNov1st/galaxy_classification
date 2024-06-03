from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from search_tool import hips_fits_url,save_fits,parse_survey,query_information
from astropy.table import Table


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Coordinate(BaseModel):
    ra:float
    dec:float
    fov:float
    size:int
    surveyid:str

# test:
# {
#     "ra": 242.274,
#     "dec":0.5721,
#     "fov":0.0116,
#     "size":400,
#     "surveyid":"P/DESI-Legacy-Surveys/DR10/color"
# }


@app.post("/coord")
async def get_coordinate(coordinate:Coordinate):
    # get coords & view info
    ra = coordinate.ra
    dec = coordinate.dec
    fov = coordinate.fov
    size = coordinate.size
    survey_id = coordinate.surveyid
    
    # get survey info:
    survey = parse_survey(survey_id)
    
    # get image:
    url,filename = hips_fits_url(ra,dec,fov,size,survey)
    print(url,filename)
    status_code = await save_fits(url,filename)
    info_table = await query_information(ra,dec)
    print(info_table)

    # return response:
    response_content = coordinate.model_dump()
    response_content['info_table'] = info_table

    return JSONResponse(content=response_content, status_code=status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)