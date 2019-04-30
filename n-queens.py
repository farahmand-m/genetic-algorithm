import numpy as np
from genetic import Genetic
from utils import plot_board, permutation_initializer


def fitness(chromosome):
    clashes = 0
    clashes += np.abs(queens - len(np.unique(chromosome)))  # Vertical Clashes
    for i in range(queens):
        for j in range(queens):
            if i != j:
                dx = np.abs(i-j)
                dy = np.abs(chromosome[i] - chromosome[j])
                if dx == dy:
                    clashes += 1
    return clashes


if __name__ == '__main__':
    queens = 8
    optimizer = Genetic(maximize=False, selection_method='tournament', tournament_size=queens**2, crossover_method='uniform', mutation_operator='scramble', mutation_probability=0.3)
    best_chromosome, _, _, _ = optimizer.run(queens, 2**queens, fitness, permutation_initializer, verbose=True, stop_on_fitness=0, stop_after_no_improvement_for=100)
    plot_board(best_chromosome, queens)
