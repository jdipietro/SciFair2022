import numpy as np
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
from tensorflow import keras
from replay_buffer import ReplayBuffer2


def build_dqn(lr, n_actions, input_dims, fc1_dim, fc2_dim):
    model = keras.Sequential()
    #    model.add(keras.Input(shape=input_dims, name="Input"))
    model.add(keras.layers.Dense(fc1_dim, activation='relu', name="layer1"))
    model.add(keras.layers.Dense(fc2_dim, activation='relu', name="layer2"))
    model.add(keras.layers.Dense(n_actions, name="output"))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model


class DeepQAgent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_decay=1.e-3, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_mode_10K.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer2(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_experience(self, state, action, reward, next_state, done):
        # always store end states
        self.memory. store_transition(state, action, reward, next_state, done)


    def choose_action(self, observation):
        if self.epsilon_end < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(next_states)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
