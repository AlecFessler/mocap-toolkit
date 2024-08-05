import copy
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataloader import HandMotionVideoDataset
from modules.ConvBlock import ConvBlock
from modules.ResBlock import ResBlock
from modules.ChannelwiseSelfAttn import ChannelwiseSelfAttn

class HandMotionPredictor(nn.Module):
    def __init__(self):
        super(HandMotionPredictor, self).__init__()
        self.fc1 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def load_model(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        model = HandMotionPredictor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def load_data(batch_size, num_workers=0):
        dataset = HandMotionVideoDataset('../hand_motion.db')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader

    @staticmethod
    def train_model(model, train_loader, test_loader, criterion, optimzer, epochs, scheduler=None, early_stopping_patience=None, trial=None):
        pass

    @staticmethod
    def evaluate(model, test_loader, criterion):
        pass

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HandMotionPredictor().to(device)

    batch_size = 1
    epochs = 100

    lr = 1e-3
    weight_decay = 1e-4
    eta_min = 1e-6

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    criterion = nn.MSELoss()

    train_loader, test_loader = HandMotionPredictor.load_data(batch_size)
    for data, labels in train_loader:
        print(f'Data shape: {data.shape}, Label count: {len(labels)}')
        break
