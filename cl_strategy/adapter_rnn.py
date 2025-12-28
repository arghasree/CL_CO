import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import json
import os
from tqdm.auto import tqdm
from model import validate
from utils.get_metrics import get_batch_cai
from utils.CONSTANTS import ORGANISMS


class AdapterLayer(nn.Module):
    """
    Adapter layer with bottleneck architecture.
    Uses residual connection for stability.
    """
    def __init__(self, input_dim, bottleneck_dim=64, activation='relu'):
        super(AdapterLayer, self).__init__()
        
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Initialize near-identity (small random weights)
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=0.01)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        """Apply adapter with residual connection"""
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Residual connection


class GRUAdapter(nn.Module):
    """
    Adapter for GRU hidden states.
    Operates on the output of BiGRU to capture species-specific sequence patterns.
    """
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super(GRUAdapter, self).__init__()
        
        self.down_project = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.Tanh()  # Use tanh for RNN outputs
        self.up_project = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Initialize near-identity
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=0.01)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        """
        Apply adapter to sequence output.
        x: (batch_size, seq_len, hidden_dim)
        """
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class RNNModelWithAdapters(nn.Module):
    """
    RNN model with adapters at both BiGRU and FC layers.
    
    Three adapter placement strategies:
    1. 'fc_only': Adapters only on FC layers (codon preferences)
    2. 'gru_only': Adapters only on BiGRU output (sequence context)
    3. 'both': Adapters on both BiGRU and FC layers (recommended)
    """
    def __init__(self, num_organisms, adapter_bottleneck_dim=64, 
                 gru_adapter_dim=128, adapter_placement='both', use_adapters=True):
        super(RNNModelWithAdapters, self).__init__()
        
        self.num_organisms = num_organisms
        self.use_adapters = use_adapters
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.gru_adapter_dim = gru_adapter_dim
        self.adapter_placement = adapter_placement  # 'fc_only', 'gru_only', 'both'
        
        # Base model
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=128, padding_idx=0)
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        
        # Organism embedding (shared)
        self.organism_embedding = nn.Embedding(num_embeddings=len(ORGANISMS), embedding_dim=32)
        
        # Base feedforward layers
        self.fc1 = nn.Linear(2*256 + 32, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.tanh_1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.tanh_2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.tanh_3 = nn.ReLU()
        
        self.fc4 = nn.Linear(128, 61)
        
        # Adapter layers (organism-specific)
        self.adapters = nn.ModuleDict()
        
        self.organism_to_idx = {}
        self.current_organism_idx = 0
        self.base_frozen = False
    
    def add_adapter(self, organism_name):
        """
        Add adapter layers for a new organism.
        Placement depends on self.adapter_placement.
        """
        if organism_name in self.adapters:
            print(f"Adapter for {organism_name} already exists!")
            return
        
        self.adapters[organism_name] = nn.ModuleDict()
        
        # GRU adapters (for sequence context)
        if self.adapter_placement in ['gru_only', 'both']:
            self.adapters[organism_name]['adapter_gru'] = GRUAdapter(
                hidden_dim=2*256,  # BiGRU output dimension
                bottleneck_dim=self.gru_adapter_dim
            ).to(next(self.parameters()).device)
        
        # FC adapters (for codon preferences)
        if self.adapter_placement in ['fc_only', 'both']:
            self.adapters[organism_name]['adapter_fc1'] = AdapterLayer(128, self.adapter_bottleneck_dim).to(next(self.parameters()).device)
            self.adapters[organism_name]['adapter_fc2'] = AdapterLayer(256, self.adapter_bottleneck_dim).to(next(self.parameters()).device)
            self.adapters[organism_name]['adapter_fc3'] = AdapterLayer(128, self.adapter_bottleneck_dim).to(next(self.parameters()).device)
        
        self.organism_to_idx[organism_name] = self.current_organism_idx
        self.current_organism_idx += 1
        
        # Count parameters
        adapter_params = sum(p.numel() for p in self.adapters[organism_name].parameters())
        print(f" Added adapter for {organism_name} ({adapter_params:,} parameters)")
    
    def freeze_base_model(self):
        """Freeze all base model parameters."""
        print("\n Freezing base model parameters...")
        
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        for param in self.bi_gru.parameters():
            param.requires_grad = False
        
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        
        for param in self.fc3.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
        
        for param in self.fc4.parameters():
            param.requires_grad = False
        
        self.base_frozen = True
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    def get_trainable_parameters(self, organism_name=None):
        """Get trainable parameters for a specific organism."""
        if organism_name is None or not self.base_frozen:
            return [p for p in self.parameters() if p.requires_grad]
        else:
            if organism_name not in self.adapters:
                raise ValueError(f"No adapter found for {organism_name}")
            
            params = []
            for param in self.adapters[organism_name].parameters():
                if param.requires_grad:
                    params.append(param)
            
            # Include organism embedding
            for param in self.organism_embedding.parameters():
                if param.requires_grad:
                    params.append(param)
            
            return params
    
    def forward(self, x, organism_id, seq_lens, organism_name=None):
        """
        Forward pass with adapters.
        
        Args:
            x: Input amino acid sequence
            organism_id: Organism ID tensor
            seq_lens: Sequence lengths
            organism_name: Name of organism (for adapter selection)
        """
        # Embedding and GRU (base model)
        x = self.embedding(x)
        packed_emb = pack_padded_sequence(x, seq_lens, batch_first=True)
        packed_output, _ = self.bi_gru(packed_emb)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_size, seq_len, 512) for BiGRU
        
        # Apply GRU adapter (if enabled)
        if (self.use_adapters and organism_name is not None and 
            organism_name in self.adapters and 
            'adapter_gru' in self.adapters[organism_name]):
            output = self.adapters[organism_name]['adapter_gru'](output)
        
        # Organism embedding
        org_emb = self.organism_embedding(organism_id)
        org_emb = org_emb.unsqueeze(1).repeat(1, output.size(1), 1)
        output = torch.cat((output, org_emb), dim=-1)
        
        # Feedforward layers with FC adapters
        batch_size, seq_len, _ = output.shape
        output = output.contiguous().view(-1, output.shape[2])
        
        # FC1 + Adapter
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.tanh_1(output)
        
        if (self.use_adapters and organism_name is not None and 
            organism_name in self.adapters and 
            'adapter_fc1' in self.adapters[organism_name]):
            output = self.adapters[organism_name]['adapter_fc1'](output)
        
        # FC2 + Adapter
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.tanh_2(output)
        
        if (self.use_adapters and organism_name is not None and 
            organism_name in self.adapters and 
            'adapter_fc2' in self.adapters[organism_name]):
            output = self.adapters[organism_name]['adapter_fc2'](output)
        
        # FC3 + Adapter
        output = self.fc3(output)
        output = self.bn3(output)
        output = self.tanh_3(output)
        
        if (self.use_adapters and organism_name is not None and 
            organism_name in self.adapters and 
            'adapter_fc3' in self.adapters[organism_name]):
            output = self.adapters[organism_name]['adapter_fc3'](output)
        
        # Final output layer
        output = self.fc4(output)
        output = output.view(batch_size, seq_len, 61)
        
        return output


def get_pad_trimmed_cds_data(cds_data, max_seq_len):
    """Trim padding from CDS data"""
    cds_data_trimmed = []
    for seq in cds_data:
        cds_data_trimmed.append(seq[0:max_seq_len])
    return torch.stack(cds_data_trimmed)


def adapter_train_epoch(model, optimizer, train_loader, loss_fn, rank, org_weights, organism_name):
    """Train for one epoch with adapters."""
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for batch in tqdm(train_loader, desc=f"Training {organism_name}"):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
        max_seq_len = max(seq_lens)
        
        aa_data_sorted = []
        cds_data_sorted = []
        for idx in range(len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[idx]])
            cds_data_sorted.append(cds_data[sorted_index[idx]])
        
        aa_data_sorted = torch.stack(aa_data_sorted)
        cds_data_sorted = torch.stack(cds_data_sorted)
        
        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank)
        organism_id = organism_id.to(rank)
        
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens, organism_name=organism_name)
        
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len)
        loss = loss_fn(output_seq_logits.permute(0, 2, 1), cds_pad_trimmed.to(rank))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
        train_cai += torch.mean(batch_cai_pred)
        train_cai_gt += torch.mean(batch_cai_gt)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_cai = train_cai / len(train_loader)
    avg_train_cai_gt = train_cai_gt / len(train_loader)
    
    return avg_train_loss, avg_train_cai, avg_train_cai_gt


def evaluate_all_tasks_adapter(model, val_loaders, org_weights, loss_fn, rank, organisms):
    """Evaluate model on all organisms using their respective adapters."""
    results = {}
    model.eval()
    
    print(f"  Evaluating on {len(organisms)} organism(s)...")
    
    with torch.no_grad():
        for org in organisms:
            def validate_with_adapter(model, loader, loss_fn, weights, rank, epoch, num_epochs):
                original_forward = model.forward
                
                def forward_with_org(*args, **kwargs):
                    return original_forward(*args, organism_name=org, **kwargs)
                
                model.forward = forward_with_org
                result = validate(model, loader, loss_fn, weights, rank, epoch, num_epochs)
                model.forward = original_forward
                return result
            
            _, val_cai, val_cai_gt = validate_with_adapter(
                model, val_loaders[org], loss_fn, 
                org_weights[org], rank, 0, 1
            )
            
            results[org] = {
                'cai': val_cai.item() if torch.is_tensor(val_cai) else val_cai,
                'cai_gt': val_cai_gt.item() if torch.is_tensor(val_cai_gt) else val_cai_gt
            }
            print(f"    {org}: CAI={results[org]['cai']:.4f}, CAI_GT={results[org]['cai_gt']:.4f}")
    
    return results


def calculate_cl_metrics(eval_matrix, organisms):
    """Calculate continual learning metrics"""
    T = len(organisms)
    R = np.zeros((T, T))
    R_gt = np.zeros((T, T))
    
    for i, train_org in enumerate(organisms):
        for j, eval_org in enumerate(organisms[:i+1]):
            R[i, j] = eval_matrix['cai'][train_org][eval_org]
            R_gt[i, j] = eval_matrix['cai_gt'][train_org][eval_org]
    
    metrics = {}
    metrics['avg_cai_final'] = float(np.mean(R[-1, :]))
    metrics['avg_cai_gt_final'] = float(np.mean(R_gt[-1, :]))
    
    forgetting = []
    forgetting_gt = []
    for j in range(T - 1):
        max_perf = np.max(R[j:, j])
        final_perf = R[-1, j]
        forgetting.append(max_perf - final_perf)
        
        max_perf_gt = np.max(R_gt[j:, j])
        final_perf_gt = R_gt[-1, j]
        forgetting_gt.append(max_perf_gt - final_perf_gt)
    
    metrics['forgetting'] = float(np.mean(forgetting)) if forgetting else 0.0
    metrics['forgetting_gt'] = float(np.mean(forgetting_gt)) if forgetting_gt else 0.0
    
    bwt = []
    bwt_gt = []
    for i in range(T - 1):
        bwt.append(R[-1, i] - R[i, i])
        bwt_gt.append(R_gt[-1, i] - R_gt[i, i])
    
    metrics['backward_transfer'] = float(np.mean(bwt)) if bwt else 0.0
    metrics['backward_transfer_gt'] = float(np.mean(bwt_gt)) if bwt_gt else 0.0
    
    metrics['learning_accuracy'] = float(np.mean(np.diag(R)))
    metrics['learning_accuracy_gt'] = float(np.mean(np.diag(R_gt)))
    metrics['plasticity'] = float(np.mean(np.diag(R)))
    metrics['plasticity_gt'] = float(np.mean(np.diag(R_gt)))
    metrics['stability'] = float(1.0 - metrics['forgetting']) if T > 1 else 1.0
    metrics['stability_gt'] = float(1.0 - metrics['forgetting_gt']) if T > 1 else 1.0
    
    metrics['final_per_task'] = {org: float(R[-1, i]) for i, org in enumerate(organisms)}
    metrics['final_per_task_gt'] = {org: float(R_gt[-1, i]) for i, org in enumerate(organisms)}
    
    return metrics, R, R_gt


def print_cl_metrics(metrics, R, R_gt, organisms, adapter_params_per_org=None):
    """Pretty print metrics with adapter-specific info"""
    print("\n" + "="*80)
    print("ADAPTER-BASED CONTINUAL LEARNING METRICS")
    print("="*80)
    
    print(f"\n OVERALL PERFORMANCE:")
    print(f"Average CAI (final):        {metrics['avg_cai_final']:.4f}")
    print(f"Average CAI_GT (final):     {metrics['avg_cai_gt_final']:.4f}")
    
    print(f"\n STABILITY & PLASTICITY:")
    print(f" Stability:    {metrics['stability']:.4f} (knowledge retention)")
    print(f"Plasticity:   {metrics['plasticity']:.4f} (learning new tasks)")
    print(f"Forgetting:   {metrics['forgetting']:.4f} (lower is better)")
    
    if adapter_params_per_org:
        print(f"\n⚡ PARAMETER EFFICIENCY:")
        total_base = adapter_params_per_org.get('base_params', 0)
        adapter_placement = adapter_params_per_org.get('adapter_placement', 'unknown')
        print(f"   Adapter placement: {adapter_placement}")
        print(f"   Base model parameters: {total_base:,}")
        for org, params in adapter_params_per_org.items():
            if org not in ['base_params', 'adapter_placement']:
                percentage = 100 * params / total_base if total_base > 0 else 0
                print(f"   {org}: {params:,} ({percentage:.2f}% of base)")
    
    print(f"\n PER-TASK PERFORMANCE:")
    for org, perf in metrics['final_per_task'].items():
        print(f"   {org:20s}: {perf:.4f}")
    
    print("="*80 + "\n")


def save_evaluation_results(eval_matrix, cl_metrics, R, R_gt, organisms, save_path, adapter_info=None):
    """Save results with adapter information"""
    os.makedirs(save_path, exist_ok=True)
    
    results = {
        'evaluation_matrix': eval_matrix,
        'cl_metrics': cl_metrics,
        'numpy_matrix_cai': R.tolist(),
        'numpy_matrix_cai_gt': R_gt.tolist(),
        'organisms': organisms,
        'method': 'adapter_based_cl',
        'adapter_info': adapter_info
    }
    
    with open(f"{save_path}/cl_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(f"{save_path}/evaluation_matrix_cai.npy", R)
    np.save(f"{save_path}/evaluation_matrix_cai_gt.npy", R_gt)
    
    print(f"\n Results saved to: {save_path}")


def train_adapter_continual_learning(train_config, train_loader, val_loader, org_weights, save_path="./results/adapter"):
    """
    Train model using adapter-based continual learning.
    
    Strategy:
    - First organism: Train entire model
    - Subsequent organisms: Freeze base, train organism-specific adapters
    
    Args:
        train_config: Dictionary containing:
            - num_epochs: Number of training epochs
            - loss_fn: Loss function
            - learning_rate: Learning rate for base model (first task)
            - adapter_learning_rate: Learning rate for adapters (subsequent tasks)
            - rank: Device (e.g., 'cuda:0')
            - organism: List of organism names
            - adapter_bottleneck_dim: Bottleneck dimension for FC adapters (default: 64)
            - gru_adapter_dim: Bottleneck dimension for GRU adapters (default: 128)
            - adapter_placement: 'fc_only', 'gru_only', or 'both' (default: 'fc_only')
        train_loader: Dictionary of training loaders
        val_loader: Dictionary of validation loaders
        org_weights: Dictionary of organism weights
        save_path: Path to save results
    
    Returns:
        loss, val_cai, val_cai_gt, evaluation_matrix, cl_metrics, model
    """
    # Extract configuration
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    base_lr = train_config.get('learning_rate', 0.01)
    adapter_lr = train_config.get('adapter_learning_rate', 0.01)
    rank = train_config['rank']
    organisms = train_config['organism']
    
    # Adapter configuration
    adapter_bottleneck_dim = train_config.get('adapter_bottleneck_dim', 64)
    gru_adapter_dim = train_config.get('gru_adapter_dim', 128)
    adapter_placement = train_config.get('adapter_placement', 'both')
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL WITH ADAPTERS")
    print("="*80)
    print(f"   Adapter placement: {adapter_placement}")
    print(f"   FC adapter bottleneck: {adapter_bottleneck_dim}")
    print(f"   GRU adapter bottleneck: {gru_adapter_dim}")
    print(f"   Number of organisms: {len(organisms)}")
    
    model = RNNModelWithAdapters(
        num_organisms=len(organisms),
        adapter_bottleneck_dim=adapter_bottleneck_dim,
        gru_adapter_dim=gru_adapter_dim,
        adapter_placement=adapter_placement,
        use_adapters=True
    )
    
    # Move model to device
    model = model.to(rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total model parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print(f"ADAPTER-BASED CONTINUAL LEARNING (Placement: {adapter_placement})")
    print("="*80)
    
    loss = {}
    val_cai = {}
    val_cai_gt = {}
    evaluation_matrix = {'cai': {}, 'cai_gt': {}}
    
    adapter_params_count = {
        'base_params': sum(p.numel() for p in model.parameters()),
        'adapter_placement': model.adapter_placement
    }
    
    for org_idx, org in enumerate(organisms):
        print("\n" + "="*80)
        print(f"Task {org_idx + 1}/{len(organisms)}: {org}")
        print("="*80)
        
        loss[org] = []
        val_cai[org] = []
        val_cai_gt[org] = []
        
        if org_idx == 0:
            print(f"Training entire model on {org} (first task)")
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        else:
            print(f"Base model frozen. Adding adapter for {org}")
            
            if not model.base_frozen:
                model.freeze_base_model()
            
            model.add_adapter(org)
            
            trainable_params = model.get_trainable_parameters(org)
            optimizer = torch.optim.Adam(trainable_params, lr=adapter_lr)
            
            adapter_params_count[org] = sum(p.numel() for p in trainable_params)
            print(f" Trainable parameters for {org}: {adapter_params_count[org]:,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            avg_loss, avg_cai, avg_cai_gt = adapter_train_epoch(
                model, optimizer, train_loader[org], loss_fn, rank, org_weights[org], org
            )
            loss[org].append(avg_loss)
            
            _, val_cai_epoch, val_cai_gt_epoch = validate(
                model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs
            )
            val_cai[org].append(val_cai_epoch)
            val_cai_gt[org].append(val_cai_gt_epoch)
            
            print(f"  Loss: {avg_loss:.4f}, Val CAI: {val_cai_epoch:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Evaluating on all {org_idx + 1} task(s)...")
        print(f"{'='*80}")
        
        tasks_seen = organisms[:org_idx + 1]
        all_task_results = evaluate_all_tasks_adapter(
            model, val_loader, org_weights, loss_fn, rank, tasks_seen
        )
        
        evaluation_matrix['cai'][org] = {}
        evaluation_matrix['cai_gt'][org] = {}
        
        for eval_org in tasks_seen:
            evaluation_matrix['cai'][org][eval_org] = all_task_results[eval_org]['cai']
            evaluation_matrix['cai_gt'][org][eval_org] = all_task_results[eval_org]['cai_gt']
    
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    
    cl_metrics, R, R_gt = calculate_cl_metrics(evaluation_matrix, organisms)
    
    print_cl_metrics(cl_metrics, R, R_gt, organisms, adapter_params_count)
    
    adapter_info = {
        'adapter_placement': model.adapter_placement,
        'adapter_bottleneck_dim': model.adapter_bottleneck_dim,
        'gru_adapter_dim': model.gru_adapter_dim,
        'num_adapters': len(model.adapters),
        'adapter_params_per_org': adapter_params_count
    }
    save_evaluation_results(evaluation_matrix, cl_metrics, R, R_gt, organisms, save_path, adapter_info)
    
    return loss, val_cai, val_cai_gt, evaluation_matrix, cl_metrics