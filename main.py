from model import RNNModel, train
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from cl_preprocessing import start_preprocessing

def run():
    model = RNNModel()
    train_config = {
        'num_epochs': 10,
        'loss_fn': CrossEntropyLoss(),
        'optimizer': optim.Adam(model.parameters(), lr=0.001),
        'rank': 'cuda:1' if __import__('torch').cuda.is_available() else 'cpu'
    }
    model.to(train_config['rank'])

    datafile_path = './cl_dataset/organism=Homo sapiens.csv'
    train_loader, val_loader, test_loader = start_preprocessing(datafile_path) 
    """
    train_loader has 661 batches where each batch has 64 sequences. Input example: tensor([11,  2, 17,  ...,  0,  0,  0])
    test_loader has 13202 batches, where each batch has 1 sequence. Input same as train_loader
    val_loader has 10562 batches, where each batch has 1 sequence. Input same as train_loader 
    """
    print("Starting training process...")
    train(train_config, model, train_loader)
    
run()