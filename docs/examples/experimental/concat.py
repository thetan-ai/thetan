"""Small script to concatenate all the excel files in a folder and compute the mean and std of each column."""

import os
import platform
from os.path import join as pjoin

import pandas as pd

if 'serges' in platform.node().lower():
    folder = '/Users/skoudoro/data/thetan_data/expdone'


files = [pjoin(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')]
df = pd.concat([pd.read_excel(f) for f in files])
result = df.groupby('Unnamed: 0').agg(['mean', 'std'])

result.to_excel(pjoin(folder, "expno_mean_std.xlsx"))