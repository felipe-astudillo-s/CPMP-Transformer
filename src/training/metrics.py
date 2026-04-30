from abc import ABC, abstractmethod
import torch

class EpochMetrics():
    def __init__(self):
        self.metrics = {}

    def add_value(self, metric_cls, value):
        if metric_cls not in self.metrics:
            self.metrics[metric_cls] = []
            
        self.metrics[metric_cls].append(value)

    def get_last_value(self, metric_cls):
        return self.metrics[metric_cls][-1]

class Metric(ABC):
    def __init__(self, name, maximize=True):
        self.name = name
        self.maximize = maximize
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, logits, y):
        pass

    @abstractmethod
    def _compute(self):
        pass

    def compute(self):
        value = self._compute()
        self.reset()
        return value

    def format(self, value):
        return f"{value:.2f}"
    
class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy")

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def step(self, logits, y):
        batch_size = y.size(0)
        # Obtenemos el índice de la predicción con mayor logit
        pred_indices = logits.argmax(dim=-1)
        
        # Verificamos si la predicción está en una posición donde y es 1
        # y[range(batch_size), pred_indices] selecciona el valor de y para la predicción hecha
        correct = y[torch.arange(batch_size), pred_indices] == 1
        
        self.total_correct += correct.sum().item()
        self.total_samples += batch_size

    def _compute(self):
        return 100 * self.total_correct / self.total_samples
    
    def format(self, value):
        return f"{value:.2f}%"
    
class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__("CrossEntropy", False)

    def reset(self):
        self.total_samples = 0
        self.total_ce = 0
    
    def step(self, logits, y):
        y = y / y.sum(dim=1, keepdim=True)
        ce = torch.nn.functional.cross_entropy(logits, y)
        batch_size = y.size(0)
        self.total_ce += ce.item() * batch_size 
        self.total_samples += batch_size
        return ce

    def _compute(self):
        return self.total_ce / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"