import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = "NewComputerModern"
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12

COLORS = ['#90C978', '#5DB1D1', '#AFD5AA', '#83C6DD', '#E8A5CB', '#E5F5A6', '#E5F5A6', '#F2E0D9', '#7840B1', 'lightgrey']

df_dcm = pd.read_csv('data/dcm_relationships.csv')
df_shap = pd.read_csv('data/shap_relationships.csv')

pink = COLORS[-6]

SHAP_intercept = 0.147081

fig, ax = plt.figure(figsize=(4.5, 3.5))
plt.scatter(df_shap.feature_value[df_shap.sex==1], df_shap.shap_effect[df_shap.sex==1], s=22, color=COLORS[0], zorder=1, edgecolor='black', linewidth=0.5, label="SHAP values $\mathrm{Sex=1}$")
plt.scatter(df_shap.feature_value[df_shap.sex==0], df_shap.shap_effect[df_shap.sex==0], s=30, marker='^', color=COLORS[1], zorder=2, edgecolor='black', linewidth=0.5, label="SHAP values $\mathrm{Sex=0}$")
plt.plot(df_dcm.Age, df_dcm.True_Male - 0.14, linewidth=0.8, color='black', zorder=-2, label="Prediction $\mathrm{Sex=1}$")
plt.plot(df_dcm.Age, df_dcm.True_Female - 0.175, '--', linewidth=0.8, color='black', zorder=-1, label="Prediction $\mathrm{Sex=0}$")
plt.ylabel('Change in absorption rate ($\Delta$mg/h)')
plt.xlabel('Age in years')
plt.legend(loc='lower right', framealpha=0)

plt.subplots_adjust(bottom=0.130, right=0.955, left=0.170, top=0.890)
plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure4.png')
plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure4.pdf')
plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure4.svg')

plt.show()