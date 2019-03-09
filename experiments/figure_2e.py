import pandas as pd
import matplotlib.pyplot as plt

SNLP = pd.read_table('./SNLP_results.tsv', header=None)
SNLP_mean = SNLP.mean(1).values
SNLP_sem = SNLP.sem(1).values
SNLP_PP = pd.read_table('./sebastian/SNLP_results_PP.tsv', header=None)
SNLP_PP_mean = SNLP_PP.mean(1).values
SNLP_PP_sem = SNLP_PP.sem(1).values
SNLP_SPGP = pd.read_table('./sebastian/SNLP_results_SPGP.tsv', header=None)
SNLP_SPGP_mean = SNLP_SPGP.mean(1).values
SNLP_SPGP_sem = SNLP_SPGP.sem(1).values

grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which SNLP, SNLP and SNLP values are computed in the paper
fullGP_SNLP = -1.3980358839035034 

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()
plt.errorbar(grid, SNLP_mean, color='r',yerr=SNLP_sem,capsize=3)
plt.errorbar(grid, SNLP_PP_mean, color='b',yerr=SNLP_PP_sem,capsize=3)
plt.errorbar(grid, SNLP_SPGP_mean, color='g',yerr=SNLP_SPGP_sem,capsize=3)
plt.plot([0,455],[fullGP_SNLP,fullGP_SNLP],color='k',linestyle='--')
#plt.ylim([-1.5,2])
plt.ylabel('SNLP')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_SNLP.pdf'
fig.savefig(filepath)
plt.close()