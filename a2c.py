import sys
import gym
import time
import pylab
import random
import numpy as np
from collections import deque
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

EPISODES = 100000

# A2C(Advantage Actor-Critic) agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load = False
        self.evaluate = False
        self.save_loc = './LunarLander_A2C'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.00001
        self.critic_lr = 0.00005

        # create model for policy network
        self.actor, self.critic, self.policy = self.build_actor_critic()

        #self.batch_size = 64
        #self.train_start = 500
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        if self.load:
            self.load_model()

    # approximate policy and value using Neural Network
    # actor: state is input and probabilities of each action is output of model
    def build_actor_critic(self):
        input = Input(shape=(self.state_size,))
        delta = Input(shape=[self.value_size])
        output1 = Dense(150, activation='relu', kernel_initializer='he_uniform')(input)
        output2 = Dense(150, activation='relu', kernel_initializer='he_uniform')(output1)
        value = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(output2)
        output3 = Dense(150, activation='relu', kernel_initializer='glorot_uniform')(output2)
        probs = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(output3)

        actor = Model(inputs=[input, delta], outputs=probs)
        critic = Model(inputs=input, outputs=value)
        policy = Model(inputs=input, outputs=probs)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*delta)

        actor.compile(loss=custom_loss, optimizer=Adam(lr=self.actor_lr))
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))

        return actor, critic, policy

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.policy.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action, policy[action]

    # update policy network every episode
    def train_model(self):
        if self.evaluate:
            return
        #batch_size = min(self.batch_size, len(self.memory))
        #mini_batch = random.sample(self.memory, batch_size)

        batch_size = len(self.memory)

        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.value_size))
        advantages = np.zeros(batch_size)
        actions = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = self.memory[i]
            value = self.critic.predict(state)[0]
            next_value = self.critic.predict(next_state)[0]

            actions[i, action] = 1
            states[i] = state
            if done:
                targets[i][0] = reward
                advantages[i] = targets[i][0] - value
            else:
                targets[i][0] = reward + self.discount_factor * next_value
                advantages[i] = targets[i][0] - value

        self.actor.fit([states, advantages], actions, epochs=1, verbose=0)
        self.critic.fit(states, targets, epochs=1, verbose=0)

    # load the saved model
    def load_model(self):
        self.actor.load_weights(self.save_loc + '_actor.h5')
        self.critic.load_weights(self.save_loc + '_critic.h5')

    # save the model which is under training
    def save_model(self):
        if not self.evaluate:
            self.actor.save_weights(self.save_loc + '_actor.h5')
            self.critic.save_weights(self.save_loc + '_critic.h5')


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

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes, filtered_scores = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        probabilities = []

        while not done:
            if agent.render:
                env.render()

            action, p = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.replay_memory(state, action, reward, next_state, done)

            score += reward
            state = next_state
            probabilities.append(p)

            if done:
                agent.train_model()

                scores.append(score)
                episodes.append(e)
                ave_score = np.mean(scores[-min(100, len(scores)):])
                filtered_scores.append(ave_score)

                pylab.gcf().clear()
                pylab.figure(figsize=(12, 8))
                pylab.plot(episodes, scores, 'b', episodes, filtered_scores, 'orange')
                pylab.savefig(agent.save_loc +  ".png")
                pylab.close()

                print("episode: {:5}   score: {:12.6}   memory length: {:4}   p: {:1.2}"
                            .format(e, ave_score, len(agent.memory), np.median(probabilities)))
                agent.memory.clear()

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if ave_score > 240:
                    np.savetxt(agent.save_loc + '.csv', filtered_scores, delimiter=",")
                    agent.save_model()
                    time.sleep(5)
                    sys.exit()

        # save the model
        if e % 100 == 0:
            agent.save_model()

    agent.save_model()
