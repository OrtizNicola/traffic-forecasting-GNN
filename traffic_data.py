# Carga de datos obtenida del codigo de pytorch geometric temporal
# con cambios menores

import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class METRLADatasetLoader(object):
    """
    Un conjunto de datos de pronóstico de tráfico basado en las 
    condiciones de tráfico del área metropolitana de Los Ángeles. 
    El conjunto de datos contiene lecturas de tráfico recogidas de 
    207 detectores de lazos en las autopistas del condado de Los 
    Ángeles, agregadas en intervalos de 5 minutos durante 4 meses 
    entre marzo de 2012 y junio de 2012.
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _download_url(self, url, save_path):  
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

        # Revisar si la carpeta zip se encuentra en la carpeta data del directorio de trabajo
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "METR-LA.zip")
        ):  
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)

        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        # Normalizamos como en el paper DCRNN (via Z-Score)
        self.means = np.mean(X, axis=(0, 2)).reshape(1, -1, 1)
        X = X - self.means
        self.stds = np.std(X, axis=(0, 2)).reshape(1, -1, 1)
        X = X / self.stds

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

        self.means = torch.tensor(self.means.squeeze()[0])
        self.stds = torch.tensor(self.stds.squeeze()[0])

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """
        
    Utiliza las características de los nodos del grafo y genera una 
    relación característica/objetivo con la forma 
    (num_nodes, num_node_features, num_timesteps_in) -> 
    (num_nodes, num_timesteps_out) 
    prediciendo la velocidad promedio del tráfico usando 
    num_timesteps_in para predecir las condiciones de tráfico en los 
    próximos num_timesteps_out.

    Args:

        num_timesteps_in (int): número de pasos temporales que el 
        modelo de secuencia observa.

        num_timesteps_out (int): número de pasos temporales que el 
        modelo de secuencia debe predecir.
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ):
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = (
            torch.tensor(self.edges), # return de adj matrix as a tensor
            self.edge_weights, 
            [i[:, 0, :] for i in self.features], # return only the speed values
            self.targets, 
            self.means, 
            self.stds # return mean and std to perform inverse transform
        )

        return dataset
