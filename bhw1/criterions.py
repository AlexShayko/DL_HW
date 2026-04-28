import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        b, n = input.shape
        squared_loss = np.sum((input - target)**2)
        return 1. / (b*n) * squared_loss

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        b, n = input.shape
        return 2./ (b*n) * (input - target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        b = input.shape[0]
        log_probs = self.log_softmax.compute_output(input)
        if self.label_smoothing == 0.0:
            mistakes = np.sum(log_probs[np.arange(b), target])
            return (-1. / b) * mistakes
        else:
            eps = float(self.label_smoothing)
            nulls = np.zeros(input.shape)
            nulls[np.arange(b), target] = 1.0
            smooth_target = nulls * (1.0 - eps) + eps / input.shape[1]
            return (-1. / b) * (np.sum(smooth_target * log_probs))

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        b = input.shape[0]
        probs = np.exp(self.log_softmax.compute_output(input))

        if self.label_smoothing == 0.0:
            nulls = np.zeros(input.shape)
            nulls[np.arange(b), target] = 1
            target_new = nulls
        else:
            eps = float(self.label_smoothing)
            nulls = np.zeros(input.shape)
            nulls[np.arange(b), target] = 1.0
            smooth_target = nulls * (1.0 - eps) + eps / input.shape[1]
            target_new = smooth_target

        return (1. / b) * (probs - target_new)