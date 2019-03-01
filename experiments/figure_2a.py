import pandas as pd
import matplotlib.pyplot as plt

KL = pd.read_table('./KL_results.tsv', header=None)
KL_mean = KL.mean(1).values
KL_std = KL.std(1).values
grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which KL, SMSE and SNLP values are computed in the paper

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()

plt.errorbar(grid, KL_mean, color='r',yerr=KL_std,capsize=5)
plt.plot([0,455],[0,0],color='k',linestyle='--')
plt.ylim([-5,75])

# save plot

filepath = 'var_KL.pdf'
fig.savefig(filepath)
plt.close()