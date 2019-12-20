import numpy as np
import matplotlib.pyplot as plt

files = ['LunarLander_DQN.csv', 'LunarLander_DoubleDQN.csv', 'LunarLander_Reinforce.csv', 'LunarLander_A2C.csv']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
for i, f in enumerate(files):
    data = np.loadtxt(f, delimiter=',')
    x = np.arange(len(data))

    if i < 2:
        x_iter = x * 64
    else:
        x_iter = x

    ax1.plot(x, data)
    ax2.plot(x_iter, data)

ax1.set_title('Score progression through training in LunarLander-v2')
ax2.set_title('Score progression through training in LunarLander-v2')

ax1.set_xlabel('Episodes Iterations')
ax2.set_xlabel('Data Interations')

ax1.set_ylabel('Score')
ax2.set_ylabel('Score')

ax1.legend(files)
ax2.legend(files)

plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.09, hspace=0.3)
plt.savefig('analysis.png')
