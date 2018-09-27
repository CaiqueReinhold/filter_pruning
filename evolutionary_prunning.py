import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf


TRAIN_STEPS = 10
LR = 0.0001
POP_SIZE = 10
MUTATION_RANGE = (0.1, 0.3)
GENERATIONS = 15
TOURNAMENT_SIZE = 1
VARIABLES_PATH = './variables/alexnet-caltech101-78'


class Individual(object):

    def __init__(self, genes, model, drop_dataset, valid_dataset,
                 variables_path):
        self.genes = genes
        self.model = model
        self.drop_dataset = drop_dataset
        self.valid_dataset = valid_dataset
        self.variables_path = variables_path
        self.drop_rate = (np.sum(self.genes) / self.genes.shape[0])

    def decode_genes(self):
        masks = tf.get_collection('MASK')
        i = 0
        dropped_filters = [set() for _ in range(len(masks))]
        for mask, dropped in zip(masks, dropped_filters):
            layer_slice = self.genes[i: mask.shape[3]]
            for index, flag in enumerate(layer_slice):
                if flag:
                    dropped.add(index)
            i += int(mask.shape[3])
        return dropped_filters

    def train(self):
        session = tf.Session()
        self._accuracy = self.model.train(
            session, self.drop_dataset, self.valid_dataset,
            lr=LR,
            epochs=TRAIN_STEPS,
            variables_path=self.variables_path,
            dropped_filters=self.decode_genes(),
            model_name=None
        )

    # def crossover(self, indv2):
    #     index = np.random.randint(len(self.genes))
    #     new_genes = np.concatenate((self.genes[:index], indv2.genes[index:]))
    #     return Individual(new_genes, self.model,
    #                       self.drop_dataset,
    #                       self.valid_dataset)

    def crossover(self, indv2):
        if self.fitness > indv2.fitness:
            indv2.genes = np.copy(self.genes)
            indv2.mutate()
            indv2.train()
        else:
            self.genes = np.copy(indv2.genes)
            self.mutate()
            self.train()

    def mutate(self):
        mut_range = np.random.uniform(*MUTATION_RANGE)
        for _ in range(int(mut_range * len(self.genes))):
            index = np.random.randint(len(self.genes))
            self.genes[index] = np.random.randint(2)

    @property
    def fitness(self):
        assert hasattr(self, '_accuracy')
        return (self.drop_rate + self._accuracy) / 2


class GeneticOptimizer(object):

    def __init__(self, pop_size, model, drop_dataset, valid_dataset,
                 variables_path, selection='tournament'):
        self.gene_size = sum(
            [int(mask.shape[3]) for mask in tf.get_collection('MASK')]
        )
        self.pop_size = pop_size
        self.model = model
        self.selection = selection
        self.drop_dataset = drop_dataset
        self.valid_dataset = valid_dataset
        self.variables_path = variables_path
        self.population = []

    def run(self):
        self.init_random_pop()

        print('random pop generated')

        for _ in range(GENERATIONS):
            print('generation {}'.format(_))

            values = []
            for indv in self.population:
                values.append((indv._accuracy, indv.drop_rate, indv.fitness))
            values.sort(key=lambda val: val[2])
            for value in values:
                print('indv acc: {0:.3f} drop: {1:.3f} fit: {2:.3f}'.format(
                      *value
                      ))

            for _ in range(self.pop_size // 2):
                indv1, indv2 = self.select()
                indv1.crossover(indv2)

        return self.population

    def init_random_pop(self):
        for _ in range(self.pop_size):
            genes = np.random.binomial(1, np.random.rand(), self.gene_size)
            indv = Individual(genes, self.model, self.drop_dataset,
                              self.valid_dataset, self.variables_path)
            indv.train()
            print('indv acc: {0:.3f} drop: {1:.3f} fit: {2:.3f}'.format(
                  indv._accuracy, indv.drop_rate, indv.fitness
                  ))
            self.population.append(indv)

    def select(self):
        method = getattr(self, 'select_{}'.format(self.selection))
        return (method(), method())

    def select_tournament(self):
        tournament = []
        for _ in range(TOURNAMENT_SIZE):
            index = np.random.randint(self.pop_size)
            tournament.append(self.population[index])
        tournament.sort(key=lambda indv: indv.fitness)
        return tournament[0]


def drop_filters(model, drop_dataset, valid_dataset, variables_path):
    optmizer = GeneticOptimizer(POP_SIZE, model, drop_dataset, valid_dataset,
                                variables_path)
    start = datetime.now()
    pop = optmizer.run()
    end = datetime.now()
    print('Elapsed time {}'.format(end - start))
    for indv in pop:
        print(indv._accuracy, indv.drop_rate)
        filename = './genes/acc_{:.3f}_drop_{:.3f}.pickle'.format(
            indv._accuracy * 100, indv.drop_rate * 100
        )
        with open(filename, 'wb') as file:
            pickle.dump(indv.genes, file)
