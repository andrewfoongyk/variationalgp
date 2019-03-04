import pandas as pd
import matplotlib.pyplot as plt

SMSE = pd.read_table('./SMSE_results.tsv', header=None)
SMSE_mean = SMSE.mean(1).values
SMSE_std = SMSE.std(1).values
grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which SMSE, SMSE and SNLP values are computed in the paper
fullGP_SMSE = 0.08689029514789581

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()
plt.errorbar(grid, SMSE_mean, color='r',yerr=SMSE_std,capsize=5)
plt.plot([0,455],[fullGP_SMSE,fullGP_SMSE],color='k',linestyle='--')
#plt.ylim([-5,75])
plt.ylabel('SMSE')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_SMSE.pdf'
fig.savefig(filepath)
plt.close()