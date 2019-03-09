import pandas as pd
import matplotlib.pyplot as plt

grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which KL, SMSE and SNLP values are computed in the paper

KL = pd.read_table('./KL_results_no_noise.tsv', header=None)
KL_mean = KL.mean(1).values
KL_sem = KL.sem(1).values
KL_PP = pd.read_table('./sebastian/KL_results_PP.tsv', header=None)
KL_PP_mean = KL_PP.mean(1).values
KL_PP_sem = KL_PP.sem(1).values
KL_SPGP = pd.read_table('./sebastian/KL_results_SPGP.tsv', header=None)
KL_SPGP_mean = KL_SPGP.mean(1).values
KL_SPGP_sem = KL_SPGP.sem(1).values

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()

plt.errorbar(grid, KL_mean, color='r',yerr=KL_sem,capsize=3)
plt.errorbar(grid,KL_PP_mean, color='b',linestyle='-',yerr=KL_PP_sem,capsize=3)
plt.errorbar(grid,KL_SPGP_mean, color='g',linestyle='-',yerr=KL_SPGP_sem,capsize=3)
plt.plot([0,455],[0,0],color='k',linestyle='--')
plt.ylim([-5,75])
plt.xlim([0,455])
plt.ylabel('KL (p||q)')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_KL.pdf'
fig.savefig(filepath)
plt.close()