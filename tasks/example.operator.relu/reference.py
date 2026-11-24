"""Reference ReLU evaluator."""


def predict(batch):
    return [[value if value > 0 else 0.0 for value in row] for row in batch]
