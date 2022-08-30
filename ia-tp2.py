import math
import random


class Event:
    def __init__(self, inputs, output):
        self.inputs = inputs  # Inputs
        self.output = output  # Expected output given inputs


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
    events = [
        Event([0, 0], 0),
        Event([0, 1], 0),
        Event([1, 0], 0),
        Event([1, 1], 1)
    ]

    weights = [
        random_weight(),
        random_weight(),
        random_weight()
    ]

    perceptron_work(events=events, weights=weights, learning_rate=0.1)

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

    weights = [
        random_weight(),
        random_weight(),
        random_weight()
    ]

    perceptron_work(events=events, weights=weights, learning_rate=0.1)


def perceptron_work(events, weights, learning_rate):
    # Agregar el 1 de control a inputs
    for event in events:
        event.inputs.append(1)

    print("inital ", weights)
    for i in range(1000):
        learn(events=events, weights=weights, learning_rate=learning_rate)
    print("final ", weights)


def learn(events, weights, learning_rate):
    for event in events:
        for index, singleInputValue in enumerate(event.inputs):
            x = singleInputValue * weights[index]
            real_output = calculate_activation(x)
            error = calculate_error(event.output, real_output)
            print("error", error)
            delta = calculate_delta(real_output, error)
            delta_weight = calculate_delta_weight(learning_rate, singleInputValue, delta)
            new_weight = calculate_adjusted_weight(weights[index], delta_weight)
            weights[index] = new_weight


def calculate_activation(input):
    return 1 / (1 + math.exp(-input))


def calculate_delta(real_output, error):
    return real_output * (1 - real_output) * error


def calculate_error(expected_output, real_output):
    return pow((expected_output - real_output), 2)


def calculate_delta_weight(learning_rate, input, delta):
    return learning_rate * input * delta


def calculate_adjusted_weight(current_weight, delta_weight):
    return current_weight + delta_weight


def random_weight():
    return random.uniform(-1, 1)


if __name__ == '__main__':
    # and_work()
    or_work()