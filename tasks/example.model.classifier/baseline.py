"""The fixed two-layer reference model."""


def predict(batch):
    hidden = [(2.0 * x - y + 1.0, x + 3.0 * y - 2.0) for x, y in batch]
    return [(left + right, 2.0 * left - right) for left, right in hidden]
