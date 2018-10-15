class Transformer:
    def __init__(self, key, transform_fn):
        self.key = key
        self.transform_fn = transform_fn

    def __call__(self, datum):
        datum[self.key] = self.transform_fn(datum[self.key])
        return datum
