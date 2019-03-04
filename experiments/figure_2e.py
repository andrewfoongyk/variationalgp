import pandas as pd
import matplotlib.pyplot as plt

SNLP = pd.read_table('./SNLP_results.tsv', header=None)
SNLP_mean = SNLP.mean(1).values
SNLP_std = SNLP.std(1).values
grid = [1,15] + list(range(30,451,30)) + [455]  # grid for which SNLP, SNLP and SNLP values are computed in the paper
fullGP_SNLP = -1.3980358839035034 

### PLOT ###

fig, ax = plt.subplots()

# plot inducing point locations

#import pdb; pdb.set_trace()
plt.errorbar(grid, SNLP_mean, color='r',yerr=SNLP_std,capsize=5)
plt.plot([0,455],[fullGP_SNLP,fullGP_SNLP],color='k',linestyle='--')
plt.ylim([-1.5,2])
plt.ylabel('SNLP')
plt.xlabel('Number of inducing points')

# save plot

filepath = 'var_SNLP.pdf'
fig.savefig(filepath)
plt.close()