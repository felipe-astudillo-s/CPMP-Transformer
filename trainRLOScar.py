                                            
from training.training import load_model, rl_train, CrossEntropyLoss, Accuracy, DataGenerationConfigRL, save_model
from generation.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV2
from generation.adapters.input.stack_features import StackFeaturesAdapterV1
from generation.adapters.output import ActionAdapter
from models.actions.cpmp_transformer_V2 import CPMPTransformer


def main():
    model = load_model(CPMPTransformer, "kaggle")

    iterations = 5
    epochs = 50
    train_size = 40000
    test_size = 10000
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 10
    seed = 43

    loss_functions = [CrossEntropyLoss()]
    metrics = [[Accuracy()]]

    datagen_config = DataGenerationConfigRL(
        instance_sets=['UC-RL-3-3', 'UC-RL-4-4', 'UC-RL-5-5', 'UC-RL-6-6', 'UC-RL-3-5', 'UC-RL-4-6', 'UC-RL-5-7', 'U$
        H=[5, 6, 7, 8, 5, 6, 7, 8, 7, 7],
        max_steps=100,
        layout_adapter_config=(EnrichedLayoutAdapter, Layout4DAdapterV2, StackFeaturesAdapterV1, 10, 8),
        moves_adapter_config=(ActionAdapter, ),
        num_workers=None
    )

    model = rl_train(model, iterations, datagen_config, epochs, train_size, test_size, batch_size, learning_rate, we$
    save_model(model, "colab_rl2")

if __name__ == "__main__":
    main()




from torch.utils.data import random_split, DataLoader
import torch
import os
import copy
import json
from settings import INSTANCE_FOLDER, MODELS_FOLDER, HYPERPARAMETERS_FOLDER
from torch.amp import GradScaler, autocast
from training.metrics import *
import random
from generation.data import generate_data_rl, split_instances
from preprocessing.dataset import load_dataset
import torch.multiprocessing as mp
import numpy as np
from utils.utils import distribuir_suma_exacta
    
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
    
def train_epoch(model, train_loader, optimizer, loss_functions, metrics_list, device, scaler):
    """
    metrics_list: Lista de listas. metrics_list[i] son las métricas para la salida i.
    """
    model.train()

    for inputs_batch, y_batch in train_loader:
        inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
        targets = [t.to(device, non_blocking=True) for t in y_batch]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type):
            logits_list = model(*inputs)
            if not isinstance(logits_list, (list, tuple)):
                logits_list = [logits_list]

            total_loss = 0
            # Iteramos por cada salida del modelo
            for i, (lf, logits, target) in enumerate(zip(loss_functions, logits_list, targets)):
                # 1. Pérdida
                total_loss += lf.step(logits, target)
                
                # 2. Métricas específicas de esta salida
                for metric in metrics_list[i]:
                    metric.step(logits, target)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Computar resultados finales de la época
    losses = [lf.compute() for lf in loss_functions]
    m_values = [[m.compute() for m in m_sublist] for m_sublist in metrics_list]
    
    return losses, m_values

def val_epoch(model, val_loader, loss_functions, metrics_list, device):
    model.eval()

    with torch.no_grad():
        for inputs_batch, y_batch in val_loader:
            inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
            targets = [t.to(device, non_blocking=True) for t in y_batch]

            logits_list = model(*inputs)
            if not isinstance(logits_list, (list, tuple)):
                logits_list = [logits_list]

            for i, (lf, logits, target) in enumerate(zip(loss_functions, logits_list, targets)):
                lf.step(logits, target)
                for metric in metrics_list[i]:
                    metric.step(logits, target)

    losses = [lf.compute() for lf in loss_functions]
    m_values = [[m.compute() for m in m_sublist] for m_sublist in metrics_list]
    
    return losses, m_values

def _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_functions, print_epoch_results, model_scorer, patience, metrics_list, device): 
    # ... (inicialización de DataLoaders y optimizer igual que antes) ...
    num_workers = os.cpu_count()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(device.type)

    train_metrics, val_metrics = EpochMetrics(), EpochMetrics()
    primary_loss = loss_functions[0]

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        train_loss_vals, train_m_vals = train_epoch(model, train_loader, optimizer, loss_functions, metrics_list, device, scaler)
        
        for lf, val in zip(loss_functions, train_loss_vals): 
            train_metrics.add_value(lf, val)
        # Añadir métricas (aplanando la lista de listas)
        for i, sublist in enumerate(train_m_vals):
            for j, val in enumerate(sublist):
                train_metrics.add_value(metrics_list[i][j], val)

        # --- VAL ---
        val_loss_vals, val_m_vals = val_epoch(model, test_loader, loss_functions, metrics_list, device)
        
        for lf, val in zip(loss_functions, val_loss_vals): 
            val_metrics.add_value(lf, val)
        for i, sublist in enumerate(val_m_vals):
            for j, val in enumerate(sublist):
                val_metrics.add_value(metrics_list[i][j], val)

        print_epoch_results(epoch, train_metrics, val_metrics)
        model_scorer.update_best_models(epoch, val_metrics)

        if epoch - model_scorer.get_last_update_epoch(primary_loss) > patience:
            print(f"Early stopping en época {epoch} (Pérdida primaria: {primary_loss.name})")
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

def config_training(model, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)
    return device

def train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_functions, patience, metrics, device):
    model_scorer = ModelScorer(model)
    primary_loss = loss_functions[0]

    def print_epoch_results(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics):
        print(f"{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs}")
        
        train_loss_str = " | ".join([f"Train {lf.name}: {lf.format(train_metrics.get_last_value(lf))}" for lf in loss_functions])
        val_loss_str = " | ".join([f"Val {lf.name}: {lf.format(val_metrics.get_last_value(lf))}" for lf in loss_functions])
        
        print(f"    {train_loss_str}")
        print(f"    {val_loss_str}")

        for i, metric in enumerate(val_metrics.metrics):
            if metric in loss_functions: 
                continue
            value = val_metrics.get_last_value(metric)
            print(f"{' | ' if i > 0 else '    '}{metric.name}: {metric.format(value)}", end='')
        print()

    _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_functions, print_epoch_results, model_scorer, patience, metrics, device)
    weights = model_scorer.get_best_weights_by_metric(primary_loss)
    model.load_state_dict(weights)
    model_scorer.print_best_score(primary_loss)

    return model

def sl_train(model, epochs, dataset, train_size, test_size, batch_size, learning_rate, weight_decay, loss_functions, patience, metrics, seed=42):
    device = config_training(model, seed)
    train_set, test_set = generate_sets(dataset, train_size, test_size, seed)
    return train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_functions, patience, metrics, device)

class DataGenerationConfigRL():
    def __init__(self, instance_sets, H, max_steps, layout_adapter_config, moves_adapter_config, num_workers):
        self.instance_sets = instance_sets
        self.H = H
        self.max_steps = max_steps
        self.layout_adapter_config = layout_adapter_config
        self.moves_adapter_config = moves_adapter_config
        self.num_workers = num_workers

def split_instances(folders, train_size, test_size, seed):
    # Mezcla aleatoria reproducible
    random.seed(seed)
    
    instance_files = []
    for instance_set in folders:
        path = INSTANCE_FOLDER / instance_set
        set_files = [os.path.join(path, f) for f in os.listdir(path)]
        random.shuffle(set_files)
        instance_files.append(set_files)

    files_len = [len(files) for files in instance_files]
    if sum(files_len) < train_size + test_size:
        train_size, test_size = distribuir_suma_exacta([train_size, test_size], sum(files_len))

    train_sizes = distribuir_suma_exacta(files_len, train_size)
    test_sizes = distribuir_suma_exacta(files_len, test_size)

    train_instances = []
    test_instances = []
    for i in range(len(instance_files)):
        train_instances.append(instance_files[i][:train_sizes[i]])
        test_instances.append(instance_files[i][train_sizes[i]:train_sizes[i] + test_sizes[i]])

    return train_instances, test_instances

def rl_train(model, iterations, datagen_config, epochs, train_size, test_size, batch_size, learning_rate, weight_decay, loss_functions, patience, metrics, seed=42):
    device = config_training(model, seed)
    train_set_file = "tmp_train.data"
    test_set_file = "tmp_test.data"
    last_avg_cost_test = None
    i = 0

    train_instances, test_instances = split_instances(datagen_config.instance_sets, train_size, test_size, seed)

    try:
        while True:
            if i > 0: print()

            mp.set_start_method('spawn', force=True)
            generate_data_rl(train_instances, 
                datagen_config.H,
                datagen_config.max_steps,
                datagen_config.layout_adapter_config,
                datagen_config.moves_adapter_config,
                model,
                batch_size,
                datagen_config.num_workers,
                output_name=train_set_file)
            
            generate_data_rl(test_instances, 
                datagen_config.H,
                datagen_config.max_steps,
                datagen_config.layout_adapter_config,
                datagen_config.moves_adapter_config,
                model,
                batch_size,
                datagen_config.num_workers,
                output_name=test_set_file)
            
            train_set = load_dataset(train_set_file, verbose=False)
            test_set = load_dataset(test_set_file, verbose=False)

            train_set._open_file()
            avg_cost_train = np.mean(train_set.file['C'])
            train_set.close()

            test_set._open_file()
            avg_cost_test = np.mean(test_set.file['C'])
            test_set.close()

            print(f"Tamaño datasets | Train: {len(train_set)} | Test: {len(test_set)}")
            print(f"Costo promedio | Train: {avg_cost_train:.2f} | Test: {avg_cost_test:.2f}")

            if last_avg_cost_test:
                current_cost_red = -(avg_cost_test - last_avg_cost_test)
                total_cost_red = -(avg_cost_test - start_avg_cost_test)
                current_gap = current_cost_red / last_avg_cost_test * 100
                total_gap = total_cost_red / start_avg_cost_test * 100

                print(f"Reducción del Costo: {current_cost_red:.2f} (acumulado {total_cost_red:.2f})")
                print(f"Reducción del Gap: {current_gap:.2f}% (acumulado {total_gap:.2f}%)")

                if avg_cost_test >= last_avg_cost_test:
                    print(f"Early stopping en iteración {i+1}")
                    break
            else:
                start_avg_cost_test = avg_cost_test

            last_avg_cost_test = avg_cost_test
            best_weights = model.state_dict()

            if i == iterations: break
            model = train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_functions, patience, metrics, device)
            i += 1

        model.load_state_dict(best_weights)
        return model
    
    finally:
        if os.path.exists(train_set_file):
            os.remove(test_set_file)
        if os.path.exists(test_set_file):
            os.remove(test_set_file)

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