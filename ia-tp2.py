import math
import random
import matplotlib.pyplot as plt
import numpy as np


historical_errors = []
historical_weights = [[] for _ in range(3)]

class Event:
    def __init__(self, inputs, output):
        self.inputs = inputs  # Inputs
        self.output = output  # Expected output given inputs

def xor_work():
    # 0 | 0 | 0
    # 0 | 1 | 0
    # 1 | 0 | 0
    # 1 | 1 | 1
    events = [
        Event([0, 0], 0),
        Event([0, 1], 1),
        Event([1, 0], 1),
        Event([1, 1], 0)
    ]

    perceptron_work(events=events)


def or_work():
    # 0 | 0 | 0
    # 0 | 1 | 0
    # 1 | 0 | 0
    # 1 | 1 | 1
    events = [
        Event([0, 0], 0),
        Event([0, 1], 1),
        Event([1, 0], 1),
        Event([1, 1], 1)
    ]

    perceptron_work(events=events)


def and_work():
    # 0 | 0 | 0
    # 0 | 1 | 0
    # 1 | 0 | 0
    # 1 | 1 | 1
    events = [
        Event([0, 0], 0),
        Event([0, 1], 0),
        Event([1, 0], 0),
        Event([1, 1], 1)
    ]

    perceptron_work(events=events)


def perceptron_work(events):
    learning_rate = 0.1

    weights = [
        0.9,
        0.66,
        -0.2
    ]

    # Agregar el 1 de control a inputs
    for event in events:
        event.inputs.insert(0, 1)  # Add at the begining the bias which is 1

    print("initial weights ", weights)
    iteration = 0
    while True:
        iteration += 1
        error = learn(events=events, weights=weights, learning_rate=learning_rate)
        print("Iteracion: ", iteration, "error: ", error)
        if abs(error) <= 0.01:
            break

    print("iteraciones", iteration)
    print("final ", weights)
    calculate_method(events, weights)


def learn(events, weights, learning_rate):
    for rowIndex, event in enumerate(events):
        x = calculate_x(event, weights)
        real_output = calculate_activation(x)
        error = event.output - real_output

        # Actualizo los pesos
        delta = calculate_delta(real_output, error)
        for index, single_input in enumerate(event.inputs):
            delta_weight = learning_rate * single_input * delta
            new_weight = weights[index] + delta_weight
            weights[index] = new_weight
            historical_weights[index].append(new_weight)
    historical_errors.append(error)
    return error


def calculate_x(event, weights):
    x = 0
    for input_index, singe_input in enumerate(event.inputs):
        x += singe_input * weights[input_index]
    return x


def calculate_activation(input):
    return 1 / (1 + math.exp(-input))


def calculate_delta(real_output, error):
    return real_output * (1 - real_output) * error

def random_weight():
    return random.uniform(-1, 1)

def calculate_method(events, weights):
    for index, event in enumerate(events):
        x = calculate_x(event, weights)
        result = calculate_activation(x)
        print("Para la entrada ", event.inputs, " la salida es ", result)

def graph_error_and_weights():
    plt.style.use('_mpl-gallery')

    # make data
    error_x = np.arange(0, len(historical_errors))
    error_y = historical_errors

    # plot
    fig, ax = plt.subplots()

    # ax.set_yscale('log')

    ax.plot(error_x, error_y, label="Error", linewidth=1.0, color="red")

    # for weights in historical_weights:
    #     weight_x = np.arange(0, len(weights))
    #     ax.plot(weight_x, weights, label="weight", linewidth=1.0, color=random.choice(["green", "yellow", "blue"]))

    plt.yticks(error_y, error_y)
    plt.xticks(range(len(error_x)), error_x)
    plt.ylabel("Erro")

    plt.show()


if __name__ == '__main__':
    #and_work()
    or_work()
    # xor_work()

    graph_error_and_weights()
