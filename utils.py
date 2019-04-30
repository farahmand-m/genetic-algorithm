import numpy as np
from matplotlib import pyplot as plt


def binary_initializer(population_size: int, chromosome_size: int):
    return np.random.randint(0, 2, (population_size, chromosome_size))


def permutation_initializer(population_size: int, chromosome_size: int):
    return np.array([np.random.permutation(chromosome_size)] * population_size)


def plot_board(permutation, queens):
    extent = (0, queens-1, 0, queens-1)
    plt.figure(frameon=False, figsize=(5, 5))
    chessboard = np.add.outer(range(queens), range(queens)) % 2
    chessboard[permutation, np.arange(queens)] = 2
    plt.imshow(chessboard, cmap='rainbow', interpolation='nearest', extent=extent, vmin=0, vmax=2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(str(permutation))
    plt.show()
