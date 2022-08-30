import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12

COLORS = ['#90C978', '#5DB1D1', '#AFD5AA', '#83C6DD', '#E8A5CB', '#E5F5A6', '#E5F5A6', '#F2E0D9', '#7840B1', 'lightgrey']

df_naive = pd.read_csv('data/result_naive_test_rmse_1.4088933124862721.csv')
df_dcm = pd.read_csv('data/result_dcm_test_rsme_1.596219679995113.csv')

t = df_dcm.TIME2[df_dcm.TIME2 > 0].values
dv = df_dcm.DV[df_dcm.TIME2 > 0].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

ax1.plot(df_naive.TIME, df_naive.PRED, linewidth=0.8, color='black', label='$\mathrm{Prediction}$')
ax1.plot(df_naive.TIME, df_naive.PRED_NO_DOSE, color=COLORS[0], label='$\mathrm{Dose}=0$')
ax1.scatter(t, dv, marker=(6, 1, 0), s=30, linewidth=0, color='black', zorder=5, label='$\mathrm{Observations}$')
ax1.set_ylabel('$\mathrm{Warfarin\ concentration\ (mg/L)}$')
ax1.set_xlabel("$\mathrm{Time\ (hours)}$")
ax1.legend(edgecolor='white')

ax2.plot(df_dcm.TIME, df_dcm.PRED, linewidth=0.8, color='black', label='$\mathrm{Prediction}$')
ax2.plot(df_dcm.TIME, df_dcm.PRED_NO_DOSE, color=COLORS[1], label='$\mathrm{Dose}=0$')
ax2.scatter(t, dv, marker=(6, 1, 0), s=30, linewidth=0, color='black', zorder=5, label='$\mathrm{Observations}$')
ax2.set_xlabel("$\mathrm{Time\ (hours)}$")
ax2.legend(edgecolor='white')

ax1.annotate('$\mathbf{A}$', (0.01, 0.94), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{B}$', (0.51, 0.94), fontsize=18, xycoords='figure fraction')

plt.subplots_adjust(top=0.895, left=0.085, bottom=0.130, right=0.975)

plt.savefig('plots/figure2.png')

plt.show()
