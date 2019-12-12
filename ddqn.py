import sys
import gym
import time
import pylab
import random
import numpy as np
import matplotlib
from collections import deque
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 10000 # max number of episodes

# this is Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see learning, then change to True
        self.render = False
        self.load = False # load an existing model
        self.evaluate = False
        self.save_loc = './LunarLander_DoubleDQN'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999750
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 500
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

        if self.load:
            self.load_model()


    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def _build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(150, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.evaluate:
            self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon and not self.evaluate:
            return np.random.randint(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def get_qvals(self, state):
        q_value = self.model.predict(state)
        return q_value

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start or self.evaluate:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    # load the saved model
    def load_model(self):
        self.model.load_weights(self.save_loc + '.h5')

    # save the model which is under training
    def save_model(self):
        if not self.evaluate:
            self.model.save_weights(self.save_loc + '.h5')

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

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, filtered_scores = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_replay()
            score += reward
            state = next_state

            if done:
                env.reset()
                # every episode update the target model to be same with model
                agent.update_target_model()


                ave_score = np.mean(scores[-min(100, len(scores)):])
                filtered_scores.append(ave_score)
                scores.append(score)
                episodes.append(e)
                pylab.gcf().clear()
                pylab.figure(figsize=(12, 8))
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig(agent.save_loc + ".png")
                pylab.close()
                print("episode: {:5}   score: {:12.6}   memory length: {:4}   epsilon {:.3}"
                            .format(e, ave_score, len(agent.memory), agent.epsilon))

                # if the mean of scores of last N episodes is bigger than X
                # stop training
                if ave_score >= 230:
                    agent.save_model()
                    time.sleep(5)   # Delays for 5 seconds. You can also use a float value.
                    sys.exit()

        # save the model every N episodes
        if e % 100 == 0:
            # TODO: save a copy of the weights
            agent.save_model()

    agent.save_model()
