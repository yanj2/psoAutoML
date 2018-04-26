import numpy as np 
from deep import train_model

def neuralnet(individual):
    # need to figure out a sensible way to put in the hyperparameter values 
    # maximise
    # Hardcoding method: kernel_size, pooling_size, dropout_rate are currently default 
    # Expecting that the first individual is ref num_conv_filters and 2nd ind is 
    # num_dense_outputs 
    num_conv_filters = int(individual[0])
    num_dense_outputs = int(individual[1])
    validation_accuracy = train_model(num_conv_outputs = num_conv_filters, 
                                      num_dense_outputs = num_dense_outputs,
                                      kernel_size = 3, 
                                      pooling_size = 2, 
                                      dropout_rate = 0.5) 
    return validation_accuracy,

def sphere(individual):
    try:
        individual = (individual - 3.)**2
        return -1.0 * (individual[0] + individual[1]),
    except:
        print(individual)

def rastrigin(individual):
    # 0.8,0.8,0.8
    sq_component = individual ** 2
    cos_component = np.cos(2 * np.pi * individual)
    summation = sq_component - 10. * cos_component
    total = 0
    for n in summation:
        total += n
    return -1.0 * (10 * DIM + total),

def ackley(individual):
    sqrt_component = np.sqrt(0.5 * np.add.reduce(np.square(individual)))
    # NOTE: converges to -1, 2 dimensions only
    cos_component = 0.5 * np.cos(2 * np.pi * individual)
    return -1.0 * (-20 * np.exp(-0.2 * sqrt_component) - np.exp(cos_component) + np.exp(1) + 20)

def rosenbrock(individual):
    summation = np.array([100*((individual[i+1] - individual[i]**2)**2) + (individual[i] - 1)**2 for i in range(len(individual)-2)])
    return -np.add.reduce(summation),

def beale(individual):
    #NOTE: 2 dimensions only
    x = individual[0]
    y = individual[1]
    first = 1.5 - x + x * y
    second = 2.25 - x + x * y ** 2
    third = 2.625 - x + x * y ** 3
    return -1.0 * (first ** 2 + second ** 2 + third ** 2),

def bukin6(individual):
    #NOTE:-15 <= x <= -5, -3 <= y <= 3
    x = individual[0]
    y = individual[1]
    sqrt_component = np.sqrt(abs(y - 0.01 * x **2))
    abs_component = 0.01 * abs(x + 10)
    return -1.0 * (100 * sqrt_component + abs_component),