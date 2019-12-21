import numpy as np
import matplotlib.pyplot as plt

off_policy_data_files = ['LunarLander_DQN.csv', 'LunarLander_DoubleDQN.csv']
off_policy_time_files = ['LunarLander_DQN_time.csv', 'LunarLander_DoubleDQN_time.csv']

on_policy_data_files = ['LunarLander_Reinforce.csv', 'LunarLander_A2C.csv']
on_policy_time_files = ['LunarLander_Reinforce_time.csv', 'LunarLander_A2C_time.csv']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
for data_f, time_f in zip(off_policy_data_files, off_policy_time_files):
    data = np.loadtxt(data_f, delimiter=',')
    time = np.loadtxt(time_f, delimiter=',')
    data_x = np.arange(len(data))

    ax1.plot(data_x, data)
    ax2.plot(time/60, data)

ax1.set_title('Off policy score progression through training in LunarLander-v2')
ax2.set_title('Off policy score progression through training in LunarLander-v2')

ax1.set_xlabel('Episodes')
ax2.set_xlabel('Time (min)')

ax1.set_ylabel('Score')
ax2.set_ylabel('Score')

ax1.legend(off_policy_data_files)
ax2.legend(off_policy_time_files)

plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.09, hspace=0.3)
plt.savefig('off_policy_analysis.png')
plt.close()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
for data_f, time_f in zip(on_policy_data_files, on_policy_time_files):
    data = np.loadtxt(data_f, delimiter=',')
    time = np.loadtxt(time_f, delimiter=',')
    data_x = np.arange(len(data))

    ax1.plot(data_x, data)
    ax2.plot(time/60, data)

ax1.set_title('On policy score progression through training in LunarLander-v2')
ax2.set_title('On policy score progression through training in LunarLander-v2')

ax1.set_xlabel('Episodes')
ax2.set_xlabel('Time (min)')

ax1.set_ylabel('Score')
ax2.set_ylabel('Score')

ax1.legend(on_policy_data_files)
ax2.legend(on_policy_time_files)

plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.09, hspace=0.3)
plt.savefig('on_policy_analysis.png')
plt.close()
