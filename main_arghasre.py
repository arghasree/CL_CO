from model_arghasre import RNNModel, train
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from cl_preprocessing import start_preprocessing
import os
from cl_strategy.ewc import train_cl_ewc
from cl_strategy.l2 import *
from cl_strategy import l2
from utils.util import loss_plot, cai_plot

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def run():
    model = RNNModel()
    train_config = {
        'num_epochs': 1,
        'loss_fn': CrossEntropyLoss(),
        'organism': ["Bacillus subtilis", "Pseudomonas putida", "Caenorhabditis elegans", "Escherichia coli general", "Saccharomyces cerevisiae"],
        'cl_strategy': 'L2', # Options: 'normal', 'EWC', 'L2'
        'dataset_dir': "./cl_dataset",
        'optimizer': optim.Adam(model.parameters(), lr=0.0001),
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
        print("Train loaders keys:", train_loader.keys())
        print("Validation loaders keys:", val_loader.keys())
        print("Test loaders keys:", test_loader.keys())
        print("Organism weights keys:", org_weights.keys())

        print("Starting CL training process using EWC strategy...")
        loss, train_cai, val_cai = train_cl_ewc(train_config, model, train_loader, val_loader, org_weights)
        loss_plot(loss, train_config['num_epochs'])
        cai_plot(val_cai, train_config['num_epochs'], len(train_config['organism']))
    elif train_config['cl_strategy'] == 'L2':
        train_config['train'] = 'first'
        train_loader = {}
        val_loader = {}
        test_loader = {}
        org_weights = {}
        for i, org in enumerate(train_config['organism']):
            print(f"\nPreparing data for organism: {org}")
            datafile_path = os.path.join(train_config['dataset_dir'], f"organism={org}.csv")
            train_l, val_l, test_l, org_w = start_preprocessing(datafile_path)
            train_loader[org] = train_l
            val_loader[org] = val_l
            test_loader[org] = test_l
            org_weights[org] = org_w
            if i == 0:
                print("Starting CL training process using L2 strategy...")
                normal_train(train_config, model, train_l, org_w, val_l)
            else:
                old_task_parameters = [p.clone().detach() for p in model.parameters()]
                train_config['old_task_parameters'] = old_task_parameters
                l2.train(train_config, model, train_l, org_w, val_l)
            
            # for testing all the tasks learned so far
            test(train_loader, val_loader, test_loader, org_weights, train_config['organism'], org, model, train_config['rank'])

def test(train_loader, val_loader, test_loader, org_weights, organisms, org, model, rank):
    results={}
    results[org]={}
    for organism in organisms:
        test_set = test_loader[organism]
        org_w = org_weights[organism]
        loss, cai, cai_gt = validate(model, test_set, org_w, rank, label=f'{organism} Test')
        results[org][organism]={}
        results[org][organism]['test']={'loss': loss, 'cai': cai, 'cai_gt': cai_gt}
        
        train_set = train_loader[organism]
        org_w = org_weights[organism]
        loss, cai, cai_gt = validate(model, train_set, org_w, rank, label=f'{organism} Train')
        results[org][organism]['train']={'loss': loss, 'cai': cai, 'cai_gt': cai_gt}
        
        val_set = val_loader[organism]
        org_w = org_weights[organism]
        loss, cai, cai_gt = validate(model, val_set, org_w, rank, label=f'{organism} Validation')
        results[org][organism]['val']={'loss': loss, 'cai': cai, 'cai_gt': cai_gt}
        
        if organism == org:
            break
        
    return results
        
        
if __name__ == "__main__":
    run()