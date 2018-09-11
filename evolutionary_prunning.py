import numpy as np
import tensorflow as tf


TRAIN_STEPS = 10
LR = 0.0001
ELITE_RETAIN = 5
POP_SIZE = 20
MUTATION_RATE = 0.5
MUTATION_RANGE = (0.2, 0.4)
GENERATIONS = 50
TOURNAMENT_SIZE = 3
VARIABLES_PATH = './variables/alexnet-caltech101-78'


class Individual(object):

    def __init__(self, genes, model, drop_dataset, valid_dataset):
        self.genes = genes
        self.model = model
        self.drop_dataset = drop_dataset
        self.valid_dataset = valid_dataset
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
            variables_path=VARIABLES_PATH,
            dropped_filters=self.decode_genes(),
            model_name=None
        )

    def crossover(self, indv2):
        index = np.random.randint(len(self.genes))
        new_genes = np.concatenate((self.genes[:index], indv2.genes[index:]))
        return Individual(new_genes, self.model,
                          self.drop_dataset,
                          self.valid_dataset)

    def mutate(self):
        mut_range = np.random.uniform(*MUTATION_RANGE)
        for _ in range(int(mut_range * len(self.genes))):
            index = np.random.randint(len(self.genes))
            self.genes[index] = 0 if self.genes[index] else 1

    @property
    def fitness(self):
        assert hasattr(self, '_accuracy')
        return (self.drop_rate + self._accuracy) / 2


class GeneticOptimizer(object):

    def __init__(self, pop_size, model, drop_dataset, valid_dataset,
                 selection='tournament'):
        self.gene_size = sum(
            [int(mask.shape[3]) for mask in tf.get_collection('MASK')]
        )
        self.pop_size = pop_size
        self.model = model
        self.selection = selection
        self.drop_dataset = drop_dataset
        self.valid_dataset = valid_dataset
        self.population = []

    def run(self):
        self.init_random_pop()

        print('random pop generated')

        for _ in range(GENERATIONS):
            new_pop = []
            print('generation {}'.format(_))

            for _ in range(self.pop_size - ELITE_RETAIN):
                indv1, indv2 = self.select()
                new_indv = indv1.crossover(indv2)
                new_pop.append(new_indv)

            for indv in new_pop:
                if MUTATION_RATE < np.random.rand():
                    indv.mutate()
                indv.train()
                print('indv acc: {0:.3f} drop: {1:.3f} fit: {2:.3f}'.format(
                    indv._accuracy, indv.drop_rate, indv.fitness
                ))

            for i in range(ELITE_RETAIN):
                new_pop.append(self.population[i])

            new_pop.sort(key=lambda indv: indv.fitness)
            self.population = new_pop

        return self.population

    def init_random_pop(self):
        for _ in range(self.pop_size):
            genes = np.random.binomial(1, np.random.rand(), self.gene_size)
            indv = Individual(genes, self.model, self.drop_dataset,
                              self.valid_dataset)
            indv.train()
            print('indv acc: {0:.3f} drop: {1:.3f} fit: {2:.3f}'.format(
                  indv._accuracy, indv.drop_rate, indv.fitness
                  ))
            self.population.append(indv)

    def select(self):
        method = getattr(self, 'select_{}'.format(self.selection))
        return method()

    def select_tournament(self):
        tournament = []
        for _ in range(TOURNAMENT_SIZE):
            index = np.random.randint(self.pop_size)
            tournament.append(self.population[index])
        tournament.sort(key=lambda indv: indv.fitness)
        return tournament[:2]


def drop_filters(model, drop_dataset, valid_dataset):
    optmizer = GeneticOptimizer(POP_SIZE, model, drop_dataset, valid_dataset)
    pop = optmizer.run()
    for indv in pop:
        print(indv._accuracy, indv.drop_rate)
