import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row', dpi=300, figsize=(2.4, 2.4))
fig.subplots_adjust(left=0.18, right=0.99, top=0.94, bottom=0.13, wspace=0.1, hspace=0.1)
# fig.tight_layout()

df = pd.read_csv("scatter_auto_mpg.csv")
ax00 = ax[0][0]
ax00.scatter(df['reward'], df['f1'], s=6, color='black', linewidth=0, alpha=0.25)
# ax00.set_xlabel('Reward', fontsize=8, labelpad=0)
ax00.set_ylabel('F1', fontsize=8, labelpad=0)
ax00.tick_params(labelsize=7)
ax00.set_title('Auto mpg', fontsize=8, pad=3)

ax10 = ax[1][0]
ax10.scatter(df['reward'], df['nhd'], s=6, color='black', linewidth=0, alpha=0.25)
ax10.set_xlabel('Reward', fontsize=8, labelpad=0)
ax10.set_ylabel('NHD', fontsize=8, labelpad=0)
ax10.tick_params(labelsize=7)

df = pd.read_csv("scatter_gaia.csv")
ax01 = ax[0][1]
ax01.scatter(df['reward'], df['f1'], s=6, color='black', linewidth=0, alpha=0.2)
# ax01.set_xlabel('Reward', fontsize=8, labelpad=0)
# ax01.set_ylabel('F1', fontsize=8, labelpad=0)
ax01.tick_params(labelsize=7)
ax01.set_title('GAIA', fontsize=8, pad=3)

ax11 = ax[1][1]
ax11.scatter(df['reward'], df['nhd'], s=6, color='black', linewidth=0, alpha=0.2)
ax11.set_xlabel('Reward', fontsize=8, labelpad=0)
# ax11.set_ylabel('NHD', fontsize=8, labelpad=0)
ax11.tick_params(labelsize=7)

# plt.show()
plt.savefig('scatter.png')
