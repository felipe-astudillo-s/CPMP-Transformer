from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from generation.adapters import LayoutDataAdapter

class Transformer(nn.Module, ABC):
    def __init__(self, layout_adapter: LayoutDataAdapter, **hyperparams):
        torch.manual_seed(42)
        super(Transformer, self).__init__()
        self.layout_adapter = layout_adapter
        self.hyperparams = hyperparams

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass