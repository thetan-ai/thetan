import os
import platform
from os.path import join as pjoin

import pandas as pd

if 'serges' in platform.node().lower():
    folder = '/Users/skoudoro/data/thetan_data/expdone'
    folder = '/Users/skoudoro/data/thetan_data/evaluations_10_v2_withFCM'
    folder = '/Users/skoudoro/data/thetan_data/evaluations_10_v2_noFCM'
    folder = '/Users/skoudoro/data/thetan_data/evaluations_10_v2'


files = [pjoin(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')]
df = pd.concat([pd.read_excel(f) for f in files])
result = df.groupby('Unnamed: 0').agg(['mean', 'std'])

result.to_excel(pjoin(folder, "expno_mean_std.xlsx"))