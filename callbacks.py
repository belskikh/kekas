class Callback:
    """
    Abstract base class used to build new callbacks.
    """
    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class PrecisionCallback(Callback):
    def __init__(self):

class PrecisionCallback(Callback):
    """
    Precision metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 precision_args: List[int] = None):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param precision_args: specifies which precision@K to log.
            [1] - accuracy
            [1, 3] - accuracy and precision@3
            [1, 3, 5] - precision at 1, 3 and 5
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.precision_args = precision_args or [1, 3, 5]

    def on_batch_end(self, state):
        prec = precision(
            state.output[self.output_key],
            state.input[self.input_key],
            topk=self.precision_args)
        for p, metric in zip(self.precision_args, prec):
            key = "precision{:02}".format(p)
            metric_ = metric.item()
            state.batch_metrics[key] = metric_