import torch
from sklearn.datasets import make_blobs
import numpy as np

from torch.utils.data import Dataset, DataLoader

class BlobsDataset(Dataset):
    def __init__(
        self, 
        n_samples_per_cluster=250, 
        n_clusters=8, 
        radius=10, 
        cluster_std=0.8
    ):
        # Generate circular centers
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)  # Equally spaced angles
        centers = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])

        # Generate blobs around these centers
        X, y = make_blobs(n_samples=n_samples_per_cluster * n_clusters, 
                        centers=centers, 
                        cluster_std=cluster_std, 
                        random_state=42)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.centers = centers
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def add_blobs(self, ax, alpha=0.7):
        ax.scatter(self.X[:, 0], self.X[:, 1], c='tab:blue', s=30, alpha=alpha)
        # ax.scatter(self.centers[:, 0], self.centers[:, 1], c='tab:red', marker='x', s=100, label='Cluster Centers')
