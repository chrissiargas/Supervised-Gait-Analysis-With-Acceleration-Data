from pre_processing.transformations import transformer
from typing import Optional, List, Tuple
import numpy as np
from math import floor


class Batcher:
    def __init__(self, S: Tuple[np.ndarray, np.ndarray], method: str, batch_size: int,
                 transform: Optional[transformer] = None):
        X, T = S
        self.data = X.copy()
        self.info = T.copy()
        self.method = method
        self.batch_size = batch_size
        self.n_wins = self.data.shape[0]
        self.length = self.data.shape[1]
        self.channels = self.data.shape[2]
        self.n_batches = floor(self.n_wins / self.batch_size)
        self.rng = np.random.default_rng(seed=1)
        self.shuffled_data = None
        self.output_data = None
        self.transform = transform
        self.shuffle = True

    def reset_data(self):
        index_list = np.arange(self.n_wins, dtype=int)

        if self.shuffle:
            self.rng.shuffle(index_list)

        self.shuffled_data = self.data[index_list]
        self.output_data = self.shuffled_data

    def __len__(self):
        return self.n_batches

    def get_shape(self):
        return self.transform.get_shape()

    def __iter__(self):
        self.reset_data()

        def gen():
            for i in range(self.n_batches):
                if self.transform:
                    batches = self.transform(self.output_data[i * self.batch_size: (i + 1) * self.batch_size])
                else:
                    batches = self.output_data[i * self.batch_size: (i + 1) * self.batch_size]

                yield batches

        return gen()


class Zipper:
    def __init__(self, batchers: List[Batcher], stack: bool = True, stack_axis: int = 0):
        self.batchers = batchers
        self.stack = stack
        self.stack_axis = stack_axis
        self.n_batches = self.batchers[0].n_batches
        assert self.n_batches == self.batchers[1].n_batches
        self.batch_size = self.batchers[0].get_shape()[0]
        self.length = self.batchers[0].get_shape()[1]
        self.channels = self.batchers[0].get_shape()[2]

    def __iter__(self):
        def gen():
            if self.stack:
                for batcher in zip(*tuple(self.batchers)):
                    yield np.stack(batcher, axis=self.stack_axis)

            else:
                for batcher in zip(*tuple(self.batchers)):
                    yield batcher

        return gen()


class Concatenator:
    def __init__(self, zippers: List[Zipper]):
        self.zippers = zippers
        self.n_dss = len(self.zippers)
        self.N_batches = sum(zipper.n_batches for zipper in zippers)
        self.batch_size = self.zippers[0].batch_size
        self.length = self.zippers[0].length
        self.channels = self.zippers[0].channels

    def get_shape(self):
        return self.N_batches, 2, self.batch_size, self.length, self.channels

    def __iter__(self):
        def gen():
            for zipper in self.zippers:
                for batch in zipper:
                    yield batch

        return gen()


def batch_concat(S: Tuple[np.ndarray, np.ndarray], method: str, batch_size: int,
                 transform: Optional[transformer] = None, same_sub: bool = True) -> Concatenator:
    X, T = S
    X = X.copy()
    T = T.copy()

    groups = []
    if same_sub:
        sub_arr = T[:, 1]
        sub_ids = np.unique(sub_arr)

        for sub_id in sub_ids:
            idx = np.argwhere(T[:, 1] == sub_id).squeeze()
            sub_X = X[idx]
            sub_T = T[idx]

            groups.append((sub_X, sub_T))
    else:
        groups.append((X, T))

    gp_batches = []
    for group in groups:
        gp_X, gp_T = group
        anchor = (gp_X[:, 0], gp_T)
        target = (gp_X[:, 1], gp_T)

        batched_anchor = Batcher(anchor, method, batch_size, transform)
        batched_target = Batcher(target, method, batch_size, transform)
        batched_datasets = [batched_anchor, batched_target]

        gp_batches.append(Zipper(batched_datasets))

    batches = Concatenator(gp_batches)

    return batches




