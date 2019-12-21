import sys
import gym
import time
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 10000 # max number of episodes


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load = False
        self.evaluate = False
        self.save_loc = './LunarLander_DQN'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 500
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load:
            self.load_model()


    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(150, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
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
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start or self.evaluate:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

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

    agent = DQNAgent(state_size, action_size)

    scores, episodes, filtered_scores, elapsed_times = [], [], [], []

    start_time = time.time()
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
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                elapsed_times.append(time.time()-start_time)
                scores.append(score)
                episodes.append(e)
                ave_score = np.mean(scores[-min(100, len(scores)):])
                filtered_scores.append(ave_score)


                pylab.gcf().clear()
                pylab.figure(figsize=(12, 8))
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig(agent.save_loc + '.png')
                pylab.close()

                print("episode: {:5}   score: {:12.6}   memory length: {:4}   epsilon {:.3}"
                            .format(e, ave_score, len(agent.memory), agent.epsilon))

                # if the mean of scores of last N episodes is bigger than X
                # stop training
                if ave_score >= 240:
                    np.savetxt(agent.save_loc + '.csv', filtered_scores, delimiter=",")
                    np.savetxt(agent.save_loc + '_time.csv', elapsed_times, delimiter=",")
                    agent.save_model()
                    time.sleep(5)   # Delays for 5 seconds. You can also use a float value.
                    sys.exit()

        # save the model
        if e % 100 == 0:
            agent.save_model()

        agent.save_model()
