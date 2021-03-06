import pandas as pd
import matplotlib.pyplot as plt

LL = pd.read_table('./LL_results_frozen.tsv', header=None)
LL_mean = LL.mean(1).values
LL_sem = LL.sem(1).values
LL_PP = pd.read_table('./sebastian/LML_results_PP.tsv', header=None)
LL_PP_mean = LL_PP.mean(1).values
LL_PP_sem = LL_PP.sem(1).values
LL_SPGP = pd.read_table('./sebastian/LML_results_SPGP.tsv', header=None)
LL_SPGP_mean = LL_SPGP.mean(1).values
LL_SPGP_sem = LL_SPGP.sem(1).values


full_LL =-140.47622680664062 
grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which LL, SMSE and SNLP values are computed in the paper

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()

plt.errorbar(grid, LL_mean, color='r',yerr=LL_sem,capsize=3)
plt.errorbar(grid, LL_PP_mean, color='b',yerr=LL_PP_sem,capsize=3)
plt.errorbar(grid, LL_SPGP_mean, color='g',yerr=LL_SPGP_sem,capsize=3)
plt.plot([0,455],[full_LL,full_LL],color='k',linestyle='--')
plt.ylim([-1500,25])
plt.xlim([0,455])
plt.yticks([-1500,-1000,-500,0])
plt.ylabel('Log marginal likelihood')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_LL_frozen.pdf'
fig.savefig(filepath)
plt.close()