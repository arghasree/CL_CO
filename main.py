from model import RNNModel, train
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from cl_preprocessing import start_preprocessing
import os
# from loss_fn_changes import *
# from cl_strategy.ewc import train_cl_ewc
from cl_strategy.ewc_with_evaluations import train_cl_ewc
from utils.util import loss_plot, cai_plot

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def run():
    model = RNNModel()
    train_config = {
        'num_epochs': 2,
        'loss_fn': CrossEntropyLoss(),
        'organism': ["Bacillus subtilis", "Pseudomonas putida", "Homo sapiens"],
        'cl_strategy': 'EWC', # Options: 'normal', 'EWC', 'L2'
        'dataset_dir': "./cl_dataset",
        'optimizer': optim.Adam(model.parameters(), lr=0.01),
        'rank': 'cuda:1' if __import__('torch').cuda.is_available() else 'cpu'
    }
        
    model.to(train_config['rank'])
    
    if train_config['cl_strategy'] == 'normal':
        datafile_path = os.path.join(train_config['dataset_dir'], f"organism={train_config['organism']}.csv")
        train_loader, val_loader, test_loader, org_weights = start_preprocessing(datafile_path) 
        """
        train_loader has 661 batches where each batch has 64 sequences. Input example: tensor([11,  2, 17,  ...,  0,  0,  0])
        test_loader has 13202 batches, where each batch has 1 sequence. Input same as train_loader
        val_loader has 10562 batches, where each batch has 1 sequence. Input same as train_loader 
        """
        print("Starting training process...")
        train(train_config, model, train_loader, org_weights, val_loader)
    elif train_config['cl_strategy'] == 'EWC':
        train_loader = {}
        val_loader = {}
        test_loader = {}
        org_weights = {}
        for org in train_config['organism']:
            print(f"Preparing data for organism: {org}")
            datafile_path = os.path.join(train_config['dataset_dir'], f"organism={org}.csv")
            train_l, val_l, test_l, org_w = start_preprocessing(datafile_path)
            train_loader[org] = train_l
            val_loader[org] = val_l
            test_loader[org] = test_l
            org_weights[org] = org_w

        print("Starting CL training process using EWC strategy...")
        loss, val_cai, val_cai_gt , _, _= train_cl_ewc(train_config, model, train_loader, val_loader, org_weights)
        print(loss)
        print(val_cai)
        print(val_cai_gt)

        # loss_plot(loss, train_config['num_epochs'])
        # cai_plot(val_cai, train_config['num_epochs'], len(train_config['organism']))
    
    elif train_config['cl_strategy'] == 'L2':
        for i, org in enumerate(train_config['organism']):
            print(f"Preparing data for organism: {org}")
            datafile_path = os.path.join(train_config['dataset_dir'], f"organism={org}.csv")
            train_l, val_l, test_l, org_w = start_preprocessing(datafile_path)
            if i == 0:
                train(train_config, model, train_l, org_w, val_l)
                old_task_parameters = [p.clone().detach() for p in model.parameters()]
            else:
                train_config['loss_fn'] = loss_cross_entropy_plus_l2
                
                

if __name__ == "__main__":
    run()