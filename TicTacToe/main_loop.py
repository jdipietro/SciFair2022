import gym
import numpy as np
import tensorflow as tf
import deep_q_network as dqn
import plot_learning as util
import game

if __name__ == '__main__':
    # Disable this feature because it is very slow
    tf.compat.v1.disable_eager_execution()
    env = game.TicTacToeGame ()
    lr = 0.001
    n_games = 10000
    agent = dqn.DeepQAgent(gamma=0.6, epsilon=1.0, lr=lr, input_dims=10,
                           n_actions=9, mem_size=1000000, batch_size=64,
                           epsilon_end=0.01)
    scores = []
    eps_history = []
    observation = env.reset()

    for i in range(n_games):
        done = False
        score = [0, 0]
        observation = env.reset()
        player = observation[9]
        while not done:
            player = observation[9]
            action = agent.choose_action(observation)
            next_observation, reward, done = env.step(action)
            score[player] += reward
            agent.store_experience(observation, action, reward, next_observation, done)
            observation = next_observation
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score[player])

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score[player],
              'average_score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        print('state: ', observation)

    graph_filename = 'nnData.png'
    x = [i + 1 for i in range(n_games)]
    util.plot_learning_curve(x, scores, eps_history, graph_filename)

    agent.save_model()