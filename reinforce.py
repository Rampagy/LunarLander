import sys
import gym
import csv
import time
import pylab
import numpy as np
from gym import wrappers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 200000 # max number of episodes

# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load = False
        self.evaluate = False
        self.record = False
        self.save_loc = './LunarLander_Reinforce'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.00002

        # create model for policy network
        self.actor, self.policy = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load:
            self.load_model()

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        input = Input(shape=(self.state_size,))
        delta = Input(shape=[self.value_size])
        output1 = Dense(300, activation='relu', kernel_initializer='he_uniform')(input)
        output2 = Dense(300, activation='relu', kernel_initializer='he_uniform')(output1)
        probs = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(output2)

        actor = Model(inputs=[input, delta], outputs=probs)
        policy = Model(inputs=input, outputs=probs)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*delta)

        actor.compile(loss=custom_loss, optimizer=Adam(lr=self.learning_rate))
        return actor, policy

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        probs = self.policy.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=probs)[0]
        return action, probs[action]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        if self.evaluate:
            return
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        actions = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            actions[i][self.actions[i]] = 1

        self.actor.fit([np.squeeze(self.states), discounted_rewards], actions, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

    # load the saved model
    def load_model(self):
        self.actor.load_weights(self.save_loc + '.h5')

    # save the model which is under training
    def save_model(self):
        if not self.evaluate:
            self.actor.save_weights(self.save_loc + '.h5')

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')

    '''
    state:
    x position
    y position
    x velocity
    y velocity
    angle
    angular velocity
    left leg contacting ground
    right leg contacting ground
    '''

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    if agent.record:
        env = wrappers.Monitor(env, agent.save_loc + '/')

    scores, episodes, filtered_scores, elapsed_times = [], [], [], []

    start_time = time.time()
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        probabilities = []

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action, p = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state
            probabilities.append(p)

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                elapsed_times.append(time.time()-start_time)
                scores.append(score)
                episodes.append(e)
                ave_score = np.mean(scores[-min(100, len(scores)):])
                filtered_scores.append(ave_score)

                if not agent.evaluate:
                    pylab.gcf().clear()
                    pylab.figure(figsize=(12, 8))
                    pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                    pylab.savefig(agent.save_loc + ".png")
                    pylab.close()
                print('episode: {:5}   score: {:12.6}   p: {:1.2}'
                            .format(e, ave_score, np.mean(probabilities)))

                # if the mean of scores of last N episodes is bigger than X
                # stop training
                if ave_score >= 240 and not agent.evaluate:
                    np.savetxt(agent.save_loc + '.csv', filtered_scores, delimiter=',')
                    np.savetxt(agent.save_loc + '_time.csv', elapsed_times, delimiter=',')
                    agent.save_model()
                    time.sleep(5)   # Delays for 5 seconds. You can also use a float value.
                    sys.exit()

        # save the model
        if e % 100 == 0:
            agent.save_model()

    agent.save_model()
