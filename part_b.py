import pandas as pd
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from adjustText import adjust_text
from pathlib import Path
import warnings
import optimization
import plotting

warnings.filterwarnings("ignore", category=FutureWarning)

# import data
data_path = str(Path().absolute()) + "/clean_data/48_Industry_Portfolios.CSV"
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2020-01-01"
N = 48
df = df.loc[df.index >= start_date, :]  # select last 5 years
# df = df / 100  # convert to percentages