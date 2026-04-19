import torch
import h5py
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.keys = list(f.attrs['key_order'])
            self.dataset_len = len(f[self.keys[0]])

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.datasets = {k: self.file[k] for k in self.keys}
        
    def __getitem__(self, idx):
        if self.file is None: self._open_file()
            
        items = []
        for k in self.keys:
            val = self.datasets[k][idx]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            else:
                val = torch.tensor(val)
            items.append(val)

        return tuple(items)
    
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __len__(self):
        return self.dataset_len

def load_dataset(filepath):
    dataset = H5Dataset(DATA_FOLDER / filepath)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data_from_path(filepath):
    with h5py.File(filepath, "r") as f:
        keys = list(f.attrs['key_order'])
        data = {k: f[k][:] for k in keys}
        data['C'] = f['C'][:]
        return data
    
def load_data(filename):
    return load_data_from_path(DATA_FOLDER / filename)

def generate_dataset(data_files, output_name, min_cost, max_cost, max_size):
    output_path = DATA_FOLDER / f"{output_name}.data"
    all_data = {}
    
    for data_file in data_files:
        path = str(DATA_FOLDER / data_file) + ".data"
        if os.path.exists(path):
            data = load_data_from_path(path) # Usa el orden correcto automáticamente
            if not all_data:
                all_data = {k: [] for k in data.keys()}
            for k in data:
                all_data[k].append(data[k])

    if not all_data: return

    key_order = [k for k in all_data.keys() if k != 'C']

    with h5py.File(output_path, "w") as f:
        f.attrs['key_order'] = key_order
        combined_data = {k: np.concatenate(all_data[k], axis=0) for k in all_data}
        
        mask = (combined_data['C'] >= min_cost) & (combined_data['C'] <= max_cost)
        final_len = min(np.sum(mask), max_size)
        
        for k in combined_data:
            f.create_dataset(k, data=combined_data[k][mask][:max_size])

    print(f"Dataset generado exitosamente en: {output_path} (Tamaño {final_len})")