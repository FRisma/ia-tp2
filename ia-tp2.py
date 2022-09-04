import math
import random


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
        0,
        0,
        0
    ]

    # Agregar el 1 de control a inputs
    for event in events:
        event.inputs.insert(0, 1)  # Add at the begining the bias which is 1

    print("initial weights ", weights)
    iteration = 0
    while True:
        iteration += 1
        error = learn(events=events, weights=weights, learning_rate=learning_rate)
        if error < 0.01:
            break

    print("iteraciones", iteration)
    print("final ", weights)
    calculate_method(events, weights)


def learn(events, weights, learning_rate):
    for event in events:
        x = calculate_x(event, weights)
        real_output = calculate_activation(x)
        error = event.output - real_output
        
        # Actualizo los pesos
        delta = calculate_delta(real_output, error)
        for index, single_input in enumerate(event.inputs):
            delta_weight = learning_rate * single_input * delta
            new_weight = weights[index] + delta_weight
            weights[index] = new_weight
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
        print("Indice", index)
        x = calculate_x(event, weights)
        result = calculate_activation(x)
        print("Para la entrada ", event.inputs, " la salida es ", result)


if __name__ == '__main__':
    #and_work()
    or_work()
    # xor_work()
