import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    #axis1 = fig.add_subplot(111, label="1")
    axis2 = fig.add_subplot(111, label="1")

 #   axis1.scatter(x, epsilons, color="blue")
  #  axis1.set_xlabel("Game", color="blue")
 #   axis1.set_ylabel("Epsilon", color="blue")
  #  axis1.tick_params(axis='x', color="blue")
 #   axis1.tick_params(axis='y', color="blue")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    axis2.plot(x, scores, color="green")
    axis2.axes.get_xaxis().set_visible(False)
    axis2.yaxis.tick_right()
    axis2.set_ylabel('Score', color="green")
    axis2.yaxis.set_label_position('right')
    axis2.tick_params(axis='y', color="green")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
