import pandas as pd
import matplotlib.pyplot as plt

KL = pd.read_table('./KL_results_no_noise.tsv', header=None)
KL_mean = KL.mean(1).values
KL_sem = KL.sem(1).values
grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which KL, SMSE and SNLP values are computed in the paper

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()

plt.errorbar(grid, KL_mean, color='r',yerr=KL_sem,capsize=3)
plt.plot([0,455],[0,0],color='k',linestyle='--')
plt.ylim([-5,75])
plt.xlim([0,455])
plt.ylabel('KL (p||q)')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_KL.pdf'
fig.savefig(filepath)
plt.close()