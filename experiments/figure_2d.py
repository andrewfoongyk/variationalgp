import pandas as pd
import matplotlib.pyplot as plt

SMSE = pd.read_table('./SMSE_results.tsv', header=None)
SMSE_mean = SMSE.mean(1).values
SMSE_sem = SMSE.sem(1).values
SMSE_PP = pd.read_table('./sebastian/SMSE_results_PP.tsv', header=None)
SMSE_PP_mean = SMSE_PP.mean(1).values
SMSE_PP_sem = SMSE_PP.sem(1).values
SMSE_SPGP = pd.read_table('./sebastian/SMSE_results_SPGP.tsv', header=None)
SMSE_SPGP_mean = SMSE_SPGP.mean(1).values
SMSE_SPGP_sem = SMSE_SPGP.sem(1).values

grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which SMSE, SMSE and SNLP values are computed in the paper
fullGP_SMSE = 0.08689029514789581

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()
plt.errorbar(grid, SMSE_mean, color='r',yerr=SMSE_sem,capsize=3)
plt.errorbar(grid, SMSE_PP_mean, color='b',yerr=SMSE_PP_sem,capsize=3)
plt.errorbar(grid, SMSE_SPGP_mean, color='g',yerr=SMSE_SPGP_sem,capsize=3)
plt.plot([0,455],[fullGP_SMSE,fullGP_SMSE],color='k',linestyle='--')
#plt.ylim([-5,75])
plt.ylabel('SMSE')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_SMSE.pdf'
fig.savefig(filepath)
plt.close()