from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import interpret
import pyearth


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = "NewComputerModern"

COLORS = ['#90C978', '#5DB1D1', '#AFD5AA', '#83C6DD', '#E8A5CB', '#E5F5A6', '#E5F5A6', '#F2E0D9', '#7840B1', 'lightgrey']

# Prep data
file = 'data/simulation_Bjorkman_advante_SHL_500.csv'
df = pd.read_csv(file)

x = (df[['WEIGHT', 'AGE']].values - df[['WEIGHT', 'AGE']].min().values) / (df[['WEIGHT', 'AGE']].max().values - df[['WEIGHT', 'AGE']].min().values)
X = np.random.normal(0, 1, (500, 48))
x = np.hstack([x, X])
y = df.CL.values

# Run models
n = df.shape[0]
k_folds = 10

lasso_coefs = np.zeros((k_folds, x.shape[1]))
rf_importance_scores = np.zeros((k_folds, x.shape[1]))
mars_importance_scores = np.zeros((k_folds, x.shape[1]))
egb_importance_scores = np.zeros((k_folds, x.shape[1]))

mars_predictions = np.zeros((k_folds, 2, n))
lasso_acc = np.zeros(k_folds)
mars_acc = np.zeros(k_folds)
rf_acc = np.zeros(k_folds)
egb_acc = np.zeros(k_folds)

x_f1 = x.copy()
x_f2 = x.copy()
x_f1[:, np.delete(np.arange(0, 50), 0)] = 0
x_f2[:, np.delete(np.arange(0, 50), 1)] = 0

for k in range(k_folds):
    print(f"Fold {k+1} ...")
    n_k = n/k_folds
    validation = np.arange(n_k * k, n_k * (k + 1), dtype=int) # For calculation of importance scores
    train = np.delete(np.arange(0, n), validation)
    x_train = x[train, :]
    y_train = y[train]
    x_validation = x[validation, :]
    y_validation = y[validation]

    # lasso = LassoCV(cv=10, fit_intercept=True).fit(x_train, y_train)
    # lasso_coefs[k, :] = lasso.coef_

    # lasso_acc[k] = np.sqrt(np.mean(np.square(y_validation - lasso.predict(x_validation))))

    # mars = pyearth.Earth(feature_importance_type='gcv') # or 'rss'?
    # mars.fit(x_train, y_train)
    # mars_importance_scores[k, :] = mars.feature_importances_
    # mars_acc[k] = np.sqrt(np.mean(np.square(y_validation - mars.predict(x_validation))))

    # for i, subset in enumerate([x_f1, x_f2]):
    #     mars_predictions[k, i, :] = mars.predict(subset)

    # rf = RandomForestRegressor().fit(x_train, y_train)

    # rf_acc[k] = np.sqrt(np.mean(np.square(y_validation - rf.predict(x_validation))))

    egb = ExplainableBoostingRegressor(interactions=0, outer_bags=150, inner_bags=150).fit(x_train, y_train)
    egb_acc[k] = np.sqrt(np.mean(np.square(y_validation - egb.predict(x_validation))))

    egb_importance_scores[k, :] = egb.explain_global().data()['scores'] / np.sum(egb.explain_global().data()['scores'])

    # # scores = permutation_importance(rf, x_validation, y_validation)['importances']
    # # rf_importance_scores[k, :] = np.mean(scores, axis=1) / np.sum(np.mean(scores, axis=1))
    # scores = rf.feature_importances_
    # rf_importance_scores[k, :] = scores / np.sum(scores)


rf_acc = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/rf_acc.npy')
mars_acc = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/mars_acc.npy')
lasso_acc = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/lasso_acc.npy')
np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/egb_acc.npy', egb_acc)

# np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/lasso_coefs.npy', lasso_coefs)
# np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/rf_importance_scores.npy', rf_importance_scores)
# np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/mars_importance_scores.npy', mars_importance_scores)
# np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/egb_importance_scores.npy', egb_importance_scores)
# np.save('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/mars_predictions.npy', mars_predictions)

lasso_coefs = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/lasso_coefs.npy')
rf_importance_scores = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/rf_importance_scores.npy')
mars_importance_scores = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/mars_importance_scores_gcv.npy')
egb_importance_scores = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/egb_importance_scores.npy')
mars_predictions = np.load('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/mars_predictions.npy')

pd.DataFrame(np.vstack([x_vis_wt, vis_wt_cl['scores'], vis_wt_cl['lower_bounds'], vis_wt_cl['upper_bounds']]).T, columns=['wt', 'score', 'lb', 'ub']).to_csv('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/checkpoints/egb_eff_wt.csv')

# LASSO
plt.bar(np.arange(x.shape[1]), np.mean(lasso_coefs, axis=0), yerr=np.std(lasso_coefs, axis=0), color=np.hstack([np.repeat(COLORS[0], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[0], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
plt.show()

# RF
plt.bar(np.arange(x.shape[1]), np.mean(rf_importance_scores, axis=0), yerr=np.std(rf_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[1], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[1], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
plt.show()

# MARS
plt.bar(np.arange(x.shape[1]), np.mean(mars_importance_scores, axis=0), yerr=np.std(mars_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[2], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[2], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
plt.show()

# EGB
plt.bar(np.arange(x.shape[1]), np.mean(egb_importance_scores, axis=0), yerr=np.std(egb_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[3], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[3], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
plt.show()

fig, ax = plt.subplots(1, 1)

plt.show()

# plot
sorter = np.argsort(mars_predictions[2, 0, :])
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7))
ax1.bar(np.arange(x.shape[1]), np.mean(lasso_coefs, axis=0), yerr=np.std(lasso_coefs, axis=0), color=np.hstack([np.repeat(COLORS[0], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[0], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
ax1.set_ylabel("$\mathrm{Coefficient\ size}$")
ax1.set_xlabel('$\mathrm{Covariate}$')

ax2.bar(np.arange(x.shape[1]), np.mean(rf_importance_scores, axis=0), yerr=np.std(rf_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[1], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[1], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
ax2.set_ylabel("$\mathrm{Normalized\ importance\ score}$")
ax2.set_xlabel('$\mathrm{Covariate}$')

ax3.bar(np.arange(x.shape[1]), np.mean(mars_importance_scores, axis=0), yerr=np.std(mars_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[2], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[2], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
ax3.set_ylabel("$\mathrm{Normalized\ importance\ score}$")
ax3.set_xlabel('$\mathrm{Covariate}$')

ax4.bar(np.arange(x.shape[1]), np.mean(egb_importance_scores, axis=0), yerr=np.std(egb_importance_scores, axis=0), color=np.hstack([np.repeat(COLORS[3], 2), np.repeat('white', 48)]), edgecolor=np.hstack([np.repeat(COLORS[3], 2), np.repeat('black', 48)]), linewidth=0.8, capsize=1, error_kw={'elinewidth': 0.8})
ax4.set_ylabel("$\mathrm{Normalized\ importance\ score}$")
ax4.set_xlabel('$\mathrm{Covariate}$')

ax5.plot(df.WEIGHT[sorter], mars_predictions[2, 0, sorter], color='black', linewidth=0.8, label='$\mathrm{MARS\ approximation}$', zorder=1)
ax5_inset = inset_axes(ax5, width="35%", height=1., loc=4)
ax5_inset.plot(df.WEIGHT[sorter], 193 * (df.WEIGHT[sorter] / 56) ** 0.8, color='black', label='$\mathrm{Ground\ truth}$', zorder=-1)
ax5_inset.set_title('$f(x)\ \mathrm{in\ model}$', {'fontsize': 12})
ax5_inset.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
ax5.scatter(44.5, 163, s=22, color=COLORS[0], zorder=2, edgecolor='black', linewidth=0.5, label="Knot location")
ax5.set_ylabel('Contribution to clearance\n prediction (mL/h)')
ax5.set_xlabel('$\mathrm{Weight}$')
ax5.legend(loc='upper left', framealpha=0)

vis_wt_cl = egb.explain_global().data(0)
x_vis_wt = np.array(vis_wt_cl['names'][:-1]) * (df.WEIGHT.max() - df.WEIGHT.min()) + df.WEIGHT.min()
ax6.plot(x_vis_wt, vis_wt_cl['scores'], color='black', linewidth=0.8)
ax6.fill_between(x_vis_wt, vis_wt_cl['lower_bounds'], vis_wt_cl['upper_bounds'], color='lightgrey', alpha=0.7)
ax6_inset = inset_axes(ax6, width="35%", height=1., loc=4)
ax6_inset.plot(df.WEIGHT[sorter], 193 * (df.WEIGHT[sorter] / 56) ** 0.8, color='black', label='$\mathrm{Ground\ truth}$', zorder=-1)
ax6_inset.set_title('$f(x)\ \mathrm{in\ model}$', {'fontsize': 12})
ax6_inset.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
ax6.set_ylabel('Explainable gradient boosting score')
ax6.set_xlabel('$\mathrm{Weight}$')

ax1.annotate('$\mathbf{A}$', (0.01, 0.95), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{B}$', (0.34, 0.95), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{C}$', (0.66, 0.95), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{D}$', (0.01, 0.46), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{E}$', (0.34, 0.46), fontsize=18, xycoords='figure fraction')
ax1.annotate('$\mathbf{F}$', (0.66, 0.46), fontsize=18, xycoords='figure fraction')

plt.subplots_adjust(bottom=0.075, top=0.935, left=0.06, right=0.970, wspace=0.300, hspace=0.320)

plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure1v2.png')
plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure1v2.pdf')
plt.savefig('/home/alexanderjanssen/PhD/Models/Pharmaceutics-review/plots/v2/figure1v2.svg')
plt.show()