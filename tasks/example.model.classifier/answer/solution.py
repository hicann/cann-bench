"""Example submission with the two affine layers folded together."""


def predict(batch):
    return [(3.0 * x + 2.0 * y - 1.0, 3.0 * x - 5.0 * y + 4.0) for x, y in batch]
