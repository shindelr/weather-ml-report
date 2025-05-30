"""
Created on Fri Jan 24 15:17:25 2025

@author: robinshindelman
@content: Functions for accessing API oriented data gathering.
"""

import requests
import pandas as pd
from sodapy import Socrata
import io

def boulder_weather_station(domain: str, token: str) -> pd.DataFrame:
    """
    API Documentation: 
        https://dev.socrata.com/foundry/data.colorado.gov/pfjr-vhp3
        
    args example: 
        domain = "data.colorado.gov"
        token = "9CRMr027a7dUWzFtpnFTukjLd"
    SoQL example: 
        where=date > 2014-12-31T00:00:00.00
    """
    client = Socrata(domain, token)
    results = client.get("pfjr-vhp3",
                         content_type='csv',
                         limit=100,
                         where="date >= '2015-01-01T00:00:00' and date < '2015-01-02T00:00:00'")
    return pd.DataFrame.from_records(results, columns=results[0])
    
def eia_query(endpoint: str, key: str):
    """
    Query the Energy Information Administration Open Data API.
    """
    params = {"api_key": key,
              "frequency": "monthly",
              "facets[stateid][]": "OR",
              "facets[sectorid][]": "RES",
              "data[0]": "sales",
              "data[1]": "customers",
              "start": "2015-01-01",
              "end": "2015-01-03",
              }
    resp = requests.get(endpoint, params=params)
    print(resp.content)
    # return pd.DataFrame.from_records(resp.content)
    

if __name__ == "__main__":
    # endpoint = f"https://api.eia.gov/v2/electricity/retail-sales/data" #"/data"
    # key = "Vt2UrTEzbZUZ0U2g7B6rKgAkcZznSNGnGpkDN7io"

    domain = "data.colorado.gov"
    token = "9CRMr027a7dUWzFtpnFTukjLd"
    df = boulder_weather_station(domain, token)
    # df = eia_query(endpoint, key)