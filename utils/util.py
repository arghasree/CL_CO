
import matplotlib.pyplot as plt
from utils.CONSTANTS import ORGANISM_DICT
def loss_plot(x, epochs):
    for t, v in x.items():
        t = ORGANISM_DICT[t]
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([f'Task {i+1}' for i in range(len(x))])
    plt.savefig('cl_loss_plot.png')

def cai_plot(x, epochs, num_task):
    for t, v in x.items():
        t = ORGANISM_DICT[t]
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)
    plt.xlabel('Epochs')
    plt.ylabel('CAI')
    plt.legend([f'Task {i+1}' for i in range(num_task)])
    plt.savefig('cl_accuracy_plot.png')
    # plt.show()