import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np

dir_error_metrics = Path('/home/florent/Projects/benchmark_satellite_coastlines/error_metrics/')
csv_error_metrics = dir_error_metrics / f'error_metrics.csv'

df = pd.read_csv(csv_error_metrics)
df['abs_bias_landsat'] =  np.abs(df['Bias Landsat (m)'])
df['abs_bias_s2'] =  np.abs(df['Bias S2(m)'])
# df.style.background_gradient(subset=["STD Land/sdsat (m)"], cmap="RdYlGn_r", vmin=0, vmax=40).to_excel("table_test.xlsx")
print(df)

# df[df['Site'] == 'DUCK'].style\
df.style\
    .background_gradient(cmap="RdYlGn_r", subset=["STD Landsat (m)", "STD S2 (m)"], vmin=5, vmax=30)\
    .background_gradient(cmap="RdYlGn_r", subset=["RMSE Landsat (m)", "RMSE S2 (m)"], vmin=6, vmax=40)\
    .background_gradient(cmap="RdYlGn", subset=["R2 Landsat", "R2 S2"], vmin=0, vmax=1)\
    .background_gradient(cmap="RdYlGn_r", subset=["abs_bias_landsat"], vmin=0, vmax=40)\
    .background_gradient(cmap="RdYlGn_r", subset=["abs_bias_s2"], vmin=0, vmax=40).highlight_null('white').to_excel("table_test.xlsx")\



# def b_g(s):
#     cm=sns.light_palette("red", as_cmap=True)
#     max_val = max(s.max(), abs(s.min()))
#     norm = colors.Normalize(0,max_val)
#     normed = norm(abs(s.values))
#     c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cm)(normed)]
#     return ['background-color: %s' % color for color in c]
#
# df.style.apply(b_g).to_excel("test.xlsx")
