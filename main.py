from model import RNNModel, train
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from cl_preprocessing import start_preprocessing
import os

def run():
    model = RNNModel()
    train_config = {
        'num_epochs': 20,
        'loss_fn': CrossEntropyLoss(),
        'organism': "Bacillus subtilis",
        'dataset_dir': "./cl_dataset",
        'optimizer': optim.Adam(model.parameters(), lr=0.0001),
        'rank': 'cuda:1' if __import__('torch').cuda.is_available() else 'cpu'
    }
    model.to(train_config['rank'])

    datafile_path = os.path.join(train_config['dataset_dir'], f"organism={train_config['organism']}.csv")
    train_loader, val_loader, test_loader, org_weights = start_preprocessing(datafile_path) 
    """
    train_loader has 661 batches where each batch has 64 sequences. Input example: tensor([11,  2, 17,  ...,  0,  0,  0])
    test_loader has 13202 batches, where each batch has 1 sequence. Input same as train_loader
    val_loader has 10562 batches, where each batch has 1 sequence. Input same as train_loader 
    """
    print("Starting training process...")
    train(train_config, model, train_loader, org_weights, val_loader)

if __name__ == "__main__":
    run()