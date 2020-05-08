import random
import numpy as np
import math


# static methods
def sigmoid(x: float) -> float:
    # print(-x)
    return 1 / (1 + math.exp(-x))


# feeds input into one layer and returns the activation of the next layer
def feed_forward(synapses: list, input_values: list, biases: list) -> list:
    temp = np.add(np.dot(synapses, input_values), biases)
    return [sigmoid(x) for x in temp]


# resets a specified number of genes from genes. Does not modify self's genes.
def mutate(genes, num_genes):
    for i in range(num_genes):
        genes[random.randint(0, len(genes) - 1)] = new_gene()

    return genes


def new_gene():
    return random.uniform(-1.2, 1.2)


class NeuralNetwork:
    def __init__(self, num_in: int, num_hidden: int, num_out: int):
        # load genes to the network
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.input_synapses = []
        self.hidden_layer_nodes = []
        self.hidden_synapses = []
        self.output_nodes = []
        self.load_genes(self.random_genes())

    # feeds input through each layer and returns list of output node activations
    def fire(self, input_values: list) -> list:
        return feed_forward(self.hidden_synapses,
                            feed_forward(self.input_synapses, input_values, self.hidden_layer_nodes),
                            self.output_nodes)

    # generate a list of random genes to load to the network
    def random_genes(self):
        length = self.num_in * self.num_hidden + self.num_hidden * self.num_out + self.num_hidden + self.num_out
        return [new_gene() for i in range(length)]

    # set the genes of the network
    def load_genes(self, genes: list):
        # layer of weights between input layer nodes and hidden layer
        # a synapse for each input node for each node in the hidden layer
        self.input_synapses = [[genes.pop(0) for i in range(self.num_in)] for i in range(self.num_hidden)]

        # middle layer of biases / nodes
        self.hidden_layer_nodes = [genes.pop(0) for i in range(self.num_hidden)]

        # layer of synapses between hidden layer and output
        self.hidden_synapses = [[genes.pop(0) for i in range(self.num_hidden)] for i in range(self.num_out)]

        # last layer of nodes
        self.output_nodes = [genes.pop(0) for i in range(self.num_out)]

    # returns a list of all of the weights and biases in the network
    def get_genes(self) -> list:
        temp = []
        for x in self.input_synapses:
            for y in x:
                temp.append(y)

        temp += self.hidden_layer_nodes

        for x in self.hidden_synapses:
            for y in x:
                temp.append(y)

        temp += self.output_nodes

        return temp

    # modifies genes in place to a combination of self.genes and mate_genes
    def mate(self, mate_genes, mutation_rate):
        if len(self.get_genes()) != len(mate_genes):
            raise ValueError('length of genes must be the same.')

        genes = self.get_genes()
        new_genes = [genes[i] if random.randint(0, 1) else mate_genes[i] for i in range(len(genes))]
        mutate(new_genes, mutation_rate)
        self.load_genes(new_genes)

    def __str__(self):
        return "input synapses: " + str(self.input_synapses) \
               + "\nhidden layer nodes: " + str(self.hidden_layer_nodes) \
               + "\nhidden layer synapses: " + str(self.hidden_synapses) \
               + "\noutput nodes: " + str(self.output_nodes)
