import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def smooth(x, y):
    smooth_x = np.linspace(x.min(), x.max(), 300)
    smooth_y = make_interp_spline(x, y)(smooth_x)
    return smooth_x, smooth_y


df = pd.read_csv("plot.csv")

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=300, figsize=(3, 2.5))
fig.subplots_adjust(left=0.15, right=0.97, top=0.94, bottom=0.13, hspace=0.3)
# fig.tight_layout()

# auto_mpg
ax0 = ax[0]
ax0.plot(*smooth(np.arange(0, 101), df['auto_mpg_mcts']), 'black', linestyle='dashed', label='MCTS')
ax0.plot(*smooth(np.arange(0, 101), df['auto_mpg_opt']), 'black', label='OPT')
# ax0.plot(np.arange(0, 101), df['auto_mpg_true'], 'black', linestyle='dotted')
ax0.set_xlim(0, 100)
ax0.set_ylim(0)
ax0.tick_params(labelsize=7)
ax0.set_title('Auto mpg', fontsize=8, pad=3)


# sachs
ax1 = ax[1]
ax1.plot(*smooth(np.arange(0, 101), df['sachs_mcts']), 'black', linestyle='dashed')
ax1.plot(*smooth(np.arange(0, 101), df['sachs_opt']), 'black')
# ax1.plot(np.arange(0, 101), df['sachs_true'], 'black', linestyle='dotted')
ax1.set_ylim(0)
ax1.tick_params(labelsize=7)
ax1.set_ylabel('Reward', fontsize=8, labelpad=1)
ax1.set_title('SACHS', fontsize=8, pad=3)


# gaia
ax2 = ax[2]
ax2.plot(*smooth(np.arange(0, 101), df['gaia_mcts']), 'black', linestyle='dashed', label='MCTS')
ax2.plot(*smooth(np.arange(0, 101), df['gaia_opt']), 'black', label='OPT')
# ax2.plot(np.arange(0, 101), df['gaia_true'], 'black', linestyle='dotted')
ax2.tick_params(labelsize=7)
ax2.set_xlabel('Number of iterations', fontsize=8, labelpad=0)
ax2.set_title('GAIA', fontsize=8, pad=3)
ax2.legend(loc='upper right', fontsize=7)


# plt.show()
plt.savefig('plot.png')
