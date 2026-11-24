"""Example ReLU submission."""


def predict(batch):
    return [[max(0.0, value) for value in row] for row in batch]
