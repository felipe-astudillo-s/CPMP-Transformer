from torch.utils.data import random_split, DataLoader
import torch
import os
import copy
import json
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER
from torch.amp import GradScaler, autocast
from training.metrics import *
import random
    
class ModelScorer:
    def __init__(self, model):
        self.model = model
        self.best_models = {}

    def update_best_models(self, epoch, val_metrics: EpochMetrics):
        for metric in val_metrics.metrics:
            sign = 1 if metric.maximize else -1
            score = sign * val_metrics.metrics[metric][-1]

            if metric in self.best_models and score < self.best_models[metric]["score"]: continue

            if metric not in self.best_models:
                self.best_models[metric] = {}
                
            self.best_models[metric]["score"] = score
            self.best_models[metric]["weights"] = copy.deepcopy(self.model.state_dict())
            self.best_models[metric]["epoch"] = epoch

    def print_best_scores(self):
        print("Mejores modelos por métrica:")
        for metric in self.best_models:
            sign = 1 if metric.maximize else -1
            print(f"    {metric.name}: {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
        
    def print_best_score(self, metric):
        sign = 1 if metric.maximize else -1
        print(f"Mejor modelo ({metric.name}): {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
    
    def get_best_weights(self):
        return {metric.name: self.best_models[metric]["weights"] for metric in self.best_models}
    
    def get_best_weights_by_metric(self, metric):
        return self.best_models[metric]["weights"]
    
    def get_last_update_epoch(self, metric):
        return self.best_models[metric]["epoch"]
    
    
def train_epoch(model, train_loader, optimizer, loss_function, metrics, device, scaler):
    model.train()

    for *inputs, y_batch in train_loader:
        inputs = [i.to(device, non_blocking=True) for i in inputs]
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Autocast para precisión mixta (FP16)
        with autocast(device.type):
            logits = model(*inputs)
            loss = loss_function.step(logits, y_batch)
            for metric in metrics: metric.step(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values

def val_epoch(model, val_loader, loss_function, metrics, device):
    model.eval()

    with torch.no_grad(), autocast(device.type):
        for batch in val_loader:
            *inputs, y_batch = [i.to(device, non_blocking=True) for i in batch]
            logits = model(*inputs)
            loss = loss_function.step(logits, y_batch)
            for metric in metrics: metric.step(logits, y_batch)

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values

def _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_function, print_epoch_results, model_scorer, patience, metrics, device): 
    num_workers = os.cpu_count()
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(device.type)

    train_metrics = EpochMetrics()
    val_metrics = EpochMetrics()

    for epoch in range(1, epochs+1):
        loss, values = train_epoch(model, train_loader, optimizer, loss_function, metrics, device, scaler)
        train_metrics.add_value(loss_function, loss)
        for i, value in enumerate(values):
            train_metrics.add_value(metrics[i], value)

        loss, values = val_epoch(model, test_loader, loss_function, metrics, device)
        val_metrics.add_value(loss_function, loss)
        for i, value in enumerate(values):
            val_metrics.add_value(metrics[i], value)

        print_epoch_results(epoch, train_metrics, val_metrics)
        model_scorer.update_best_models(epoch, val_metrics)

        if epoch - model_scorer.get_last_update_epoch(loss_function) > patience:
            print("Early stopping en época", epoch)
            break

    return train_metrics, val_metrics

def generate_sets(dataset, train_size, test_size, seed):
    generator = torch.Generator().manual_seed(seed)
    remaining_size = len(dataset) - train_size - test_size

    train_set, test_set, _ = random_split(
        dataset, 
        [train_size, test_size, remaining_size],
        generator=generator
    )

    return train_set, test_set

def train(model, epochs, dataset, train_size, test_size, batch_size, learning_rate, weight_decay, patience, metrics, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    train_set, test_set = generate_sets(dataset, train_size, test_size, seed)

    ### CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)
    
    loss_function = CrossEntropyLoss()
    model_scorer = ModelScorer(model)

    def print_epoch_results(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics):
        print(f'{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs}')
        print(f"    Average - Train Loss: {loss_function.format(train_metrics.get_last_value(loss_function))}", end='')
        print(f" | Val Loss: {loss_function.format(val_metrics.get_last_value(loss_function))}")

        for i, metric in enumerate(val_metrics.metrics):
            if metric == loss_function: continue
            value = val_metrics.get_last_value(metric)
            print(f'{' | ' if i > 0 else '    '}{metric.name}: {metric.format(value)}', end='')
        print()

    train_metrics, val_metrics = _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_function, print_epoch_results, model_scorer, patience, metrics, device)
    weights = model_scorer.get_best_weights_by_metric(loss_function)
    model.load_state_dict(weights)
    model_scorer.print_best_score(loss_function)

    return model

def save_model(model, model_name):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    print(f"✅ Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model