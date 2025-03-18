"""
author: Robin Shindelman
date: 2025-02-27
description: Data processing for rain_or_norain.csv.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_csv(data_fp: str) -> pd.DataFrame:
    """ Load the .csv into a panda dataframe """
    return pd.read_csv(data_fp)

data_fp = '/Users/robinshindelman/repos/332-project/data/raw/rain/rain_or_norain.csv'
df = load_csv(data_fp)
