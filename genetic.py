import numpy as np


class Genetic:

    def __init__(self, maximize=True, selection_method='fitness', tournament_size=None, crossover_method='one-point', crossover_probability=0.8, mutation_operator='bit-flip', mutation_probability=0.2, survivor_selection='fitness', elitism=False):
        """
        :param selection_method: Possible values are "fitness", and "tournament"
        :param crossover_method: Possible values are "one-point", and "uniform"
        :param mutation_operator: Possible values are "bit-flip", "swap", and "scramble"
        :param survivor_selection: Right now the only possible method is "fitness" :(
        """
        self.maximize = maximize
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.crossover_probability = crossover_probability
        self.mutation_operator = mutation_operator
        self.mutation_probability = mutation_probability
        self.survivor_selection = survivor_selection
        self.elitism = elitism

    def select(self, population, fitness_values):
        if self.selection_method == 'fitness':
            chances = fitness_values + 2 * np.abs(np.min(fitness_values))
            chances = chances if self.maximize else (1 / chances)
            chances = chances / np.sum(chances)
            chosen_ones = np.random.choice(np.arange(len(population)), 2, p=chances)
            return population[chosen_ones]
        if self.selection_method == 'tournament':
            if self.tournament_size is None:
                raise ValueError('Tournament Size has not been specified')
            selected = []
            for i in range(2):
                candidates = np.random.choice(np.arange(len(population)), self.tournament_size)
                candidates_fitness = fitness_values[candidates]
                selected.append(candidates[np.argmax(candidates_fitness) if self.maximize else np.argmin(candidates_fitness)])
            return population[selected]
        raise ValueError('Invalid selection method')

    def crossover(self, parents):
        if np.random.rand() <= self.crossover_probability:
            if self.crossover_method == 'one-point':
                point = np.random.randint(parents.shape[1])
                return np.array([np.append(parents[0, :point], parents[1, point:]), np.append(parents[1, :point], parents[0, point:])])
            if self.crossover_method == 'uniform':
                random_values = np.random.randint(0, 2, parents.shape[1])
                first_child = [parents[random_values[i]][i] for i in range(parents.shape[1])]
                second_child = [parents[1-random_values[i]][i] for i in range(parents.shape[1])]
                return np.array([first_child, second_child])
            raise ValueError('Invalid crossover method')
        else:
            return parents

    def mutate(self, chromosome):
        if np.random.rand() <= self.mutation_probability:
            mutated = chromosome.copy()
            if self.mutation_operator == 'bit-flip':
                random_bit = np.random.randint(len(chromosome))
                np.put(mutated, random_bit, 1-chromosome[random_bit])
                return mutated
            if self.mutation_operator == 'swap':
                random_indexes = np.random.randint(len(chromosome), size=2)
                mutated[random_indexes[0]], mutated[random_indexes[1]] = mutated[random_indexes[1]], mutated[random_indexes[0]]
                return mutated
            if self.mutation_operator == 'scramble':
                start, end = tuple(sorted(np.random.randint(len(chromosome), size=2)))
                np.random.shuffle(mutated[start:end])
                return mutated
            raise ValueError('Invalid mutation operator')
        else:
            return chromosome

    def find_survivors(self, mixed_population, fitness_values):
        if self.survivor_selection == 'fitness':
            chances = fitness_values + 1
            chances = chances if self.maximize else (1 / chances)
            chances = chances / np.sum(chances)
            chosen_ones = np.random.choice(np.arange(len(mixed_population)), size=mixed_population.shape[0] // 2, p=chances)
            if chances.argmax() not in chosen_ones and self.elitism:
                np.put(chosen_ones, chances[chosen_ones].argmin(), chances.argmax())
            return mixed_population[chosen_ones]
        raise ValueError('Invalid survival selection method')

    def run(self, chromosome_size: int, population_size: int, fitness_function, initializer, verbose=False, maximum_epochs=1000, stop_after_no_improvement_for=None, stop_on_fitness=None):
        population = initializer(population_size, chromosome_size)
        population_fitness = np.array([fitness_function(chromosome) for chromosome in population])
        best_previous_chromosome_fitness = population_fitness.max() if self.maximize else population_fitness.min()
        best_initial_fitness = best_previous_chromosome_fitness
        best_fitness_values_over_epochs = [best_previous_chromosome_fitness]
        epochs_without_improvement = 0
        for epoch in range(maximum_epochs):
            new_population = []
            for i in range(population.shape[0] // 2):
                parents = self.select(population, population_fitness)
                children = self.crossover(parents)
                children[0] = self.mutate(children[0])
                new_population.append(children[0])
                children[1] = self.mutate(children[1])
                new_population.append(children[1])
            mixed_population = np.append(population, np.array(new_population), 0)
            mixed_population_fitness = np.array([fitness_function(chromosome) for chromosome in mixed_population])
            new_population = self.find_survivors(mixed_population, mixed_population_fitness)
            new_population_fitness = np.array([fitness_function(chromosome) for chromosome in new_population])
            best_new_chromosome_fitness = new_population_fitness.max() if self.maximize else new_population_fitness.min()
            change = best_new_chromosome_fitness - best_previous_chromosome_fitness
            if (self.maximize and change > 0) or (not self.maximize and change < 0):
                experienced_no_improvement = False
                epochs_without_improvement = 0
            else:
                experienced_no_improvement = True
                epochs_without_improvement += 1
            if verbose:
                print('Epoch {}: Best fitness value: {:.2f} {}'.format(epoch, best_new_chromosome_fitness, '(No improvement)' if experienced_no_improvement else ''))
            population = new_population
            population_fitness = new_population_fitness
            best_previous_chromosome_fitness = best_new_chromosome_fitness
            best_fitness_values_over_epochs.append(best_previous_chromosome_fitness)
            if stop_after_no_improvement_for is not None:
                if epochs_without_improvement >= stop_after_no_improvement_for:
                    print('No improvement experienced for {} epochs. Ending the optimization procedure'.format(epochs_without_improvement))
                    break
            if stop_on_fitness is not None:
                if best_previous_chromosome_fitness == stop_on_fitness:
                    print('Fitness value {} achieved. Ending the optimization procedure'.format(best_previous_chromosome_fitness))
                    break
        best = np.argmax(population_fitness) if self.maximize else np.argmin(population_fitness)
        return population[best], population_fitness[best], best_initial_fitness, best_fitness_values_over_epochs
