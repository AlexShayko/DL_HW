import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ

        return int(np.ceil(self.X.shape[0]/self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ

        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        self.batch_id = 0
        if self.shuffle:
            self.indices = np.random.permutation(len(self.X))
        else:
            self.indices = np.arange(len(self.X))
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        if self.batch_id * self.batch_size >= len(self.X):
            raise StopIteration

        batch_now_start = self.batch_id * self.batch_size
        batch_now_end = (self.batch_id + 1) * self.batch_size

        batch_X = self.X[self.indices[batch_now_start:batch_now_end]]
        batch_Y = self.y[self.indices[batch_now_start:batch_now_end]]

        self.batch_id += 1

        return batch_X, batch_Y



