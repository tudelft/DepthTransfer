from typing import Tuple
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device: str = "cpu"):
        """
        Calculates the running mean and std of a data stream using PyTorch.
        This is useful when dealing with streaming data in machine learning scenarios.

        :param epsilon: A small number to avoid division by zero.
        :param shape: The shape of the data stream's outputs.
        """
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        Creates a deep copy of the current object.

        :return: A new instance of RunningMeanStd with the same data.
        """
        new_object = RunningMeanStd(epsilon=self.count - 1e-4, shape=self.mean.shape)
        new_object.mean = self.mean.clone()
        new_object.var = self.var.clone()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another RunningMeanStd object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        """
        Update the running mean and variance with new data.

        :param arr: New data as a PyTorch tensor.
        """
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        """
        Update from precomputed mean and variance.

        :param batch_mean: Computed mean of the batch.
        :param batch_var: Computed variance of the batch.
        :param batch_count: Number of elements in the batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
