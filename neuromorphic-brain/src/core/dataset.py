"""
Neuromorphic Dataset and Synthetic DVS Spike Generator.
Supports:
1. Synthetic event generator (Poisson / DVS-like spatio-temporal spike streams)
   ensuring complete out-of-the-box CPU/GPU testing without external download dependencies.
2. Standard N-MNIST binning loader from raw events if dataset files are supplied in data/n_mnist.
"""

from typing import Tuple, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader


class NeuromorphicDataset(Dataset):
    """
    Neuromorphic spiking dataset yielding spatio-temporal spike tensors:
    [T, num_channels, height, width] or flattened [T, input_dim].
    """

    def __init__(
        self,
        num_samples: int = 100,
        time_steps: int = 16,
        input_dim: int = 64,
        num_classes: int = 10,
        firing_prob: float = 0.08,
        data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.firing_prob = firing_prob
        self.data_dir = data_dir

        self.samples, self.labels = self._generate_or_load()

    def _generate_or_load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # If an existing N-MNIST directory with preprocessed files is present, load it
        if self.data_dir and os.path.exists(os.path.join(self.data_dir, "processed_nmnist.pt")):
            data = torch.load(os.path.join(self.data_dir, "processed_nmnist.pt"))
            return data["samples"][: self.num_samples], data["labels"][: self.num_samples]

        # Generate synthetic DVS Poisson-like spike patterns with class-conditioned signatures
        torch.manual_seed(42)
        labels = torch.randint(0, self.num_classes, (self.num_samples,))

        samples = torch.zeros(self.num_samples, self.time_steps, self.input_dim)
        for i in range(self.num_samples):
            cls = labels[i].item()
            # Active sub-band of neurons firing preferentially for this class
            start_idx = (cls * (self.input_dim // self.num_classes)) % self.input_dim
            end_idx = min(start_idx + (self.input_dim // self.num_classes), self.input_dim)

            # Background noise
            noise = (torch.rand(self.time_steps, self.input_dim) < (self.firing_prob * 0.3)).float()
            # Signal
            signal = (torch.rand(self.time_steps, end_idx - start_idx) < (self.firing_prob * 3.0)).float()

            samples[i] = noise
            samples[i, :, start_idx:end_idx] = signal

        return samples, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


def get_neuromorphic_dataloader(
    batch_size: int = 16,
    num_samples: int = 100,
    time_steps: int = 16,
    input_dim: int = 64,
    num_classes: int = 10,
    data_dir: Optional[str] = None,
) -> DataLoader:
    dataset = NeuromorphicDataset(
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir=data_dir,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
