import argparse
import torch
import os
# import torch.nn as nn
import pandas as pd
import wandb
from tqdm.auto import tqdm as tqdm
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.multiprocessing as mp
from models.RNN import RNNModel
from data_preprocessing import start_preprocessing
from utils.custom_loss import CAI_LOSS, CrossEntropyLoss, SoftmaxCAI, CaiMseLoss, StabilityMseLoss
from utils.util import get_batch_cai, get_batch_stability, get_batch_gc_content


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# wandb.login(key = "cddbb81e657d85514600791c422ff35c68117a53")

rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(rank)
print(rank)


class Train:
    def __init__(self, dataset_path, cai_type, required_cai, required_stability, stability_type='deg', alpha=0.5, temperature=37, tool_pkg='rnasoft'):
        self.dataset_path = dataset_path
        # self.alpha = alpha
        # self.cai_type = cai_type
        # self.stability_type = stability_type
        # self.pkg = tool_pkg
        # self.T = temperature
        # self.required_cai = required_cai 
        # self.required_stability = required_stability
        # self.bta = 0.1

    def get_token_dict_and_DataLoader(self):
        train_loader, val_loader, test_loader, cds_token_dict = start_preprocessing(self.dataset_path)
        return train_loader, val_loader, test_loader, cds_token_dict
    
    
    def get_pad_trimmed_cds_data(self, cds_data, max_seq_len):
        # Trim till max_len 
        # Trims padded portion 
        cds_data_trimmed = []
        for id, seq in enumerate(cds_data):
            # print(type(seq))
            cds_data_trimmed.append(seq[0:max_seq_len])
        
        cds_data_trimmed = torch.stack(cds_data_trimmed)
        return cds_data_trimmed

    def get_padded_output(self,output_seq_logits, seq_lens):
        # used for encoding 
        for i in range(0, len(seq_lens)):
            output_seq_logits[i][seq_lens[i]:][:] = -100
        
        return output_seq_logits
        
    # def get_correct_tags(self, output_seq_logits, cds_data, seq_lens):
    #     right_token =0
    #     total_tokens = 0
    #     for i in range(0, len(seq_lens)):
    #         seq_i = torch.argmax(output_seq_logits[i], dim=-1)
    #         # Excluding Start and Stop codons
    #         seq_i = seq_i[:seq_lens[i]]
    #         cds_data_i = cds_data[i][:seq_lens[i]]
    #         total_tokens += len(cds_data_i)
    #         # get token wise accuracy
    #         for j in range(0, len(seq_i)):
    #             if seq_i[j] == cds_data_i[j]:
    #                 right_token += 1
        
    #     return right_token, total_tokens
    
    def train_validate_model(self, rank, model, train_config, train_loader, val_loader, optimizer, cross_entropy_loss, cds_token_dict):
        
        train_losses_epoch = []
        val_losses_epoch = []
        l1_train_epoch = []
        l2_train_epoch = []
        l1_val_epoch = []
        l2_val_epoch = []
        train_accuracy_epoch = []
        val_accuracy_epoch = []

        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience = train_config['patience']
        patience_counter = 0


        num_epochs = train_config['num_epochs']
        
        # wandb.watch(model, cross_entropy_loss, log=None, log_freq=20)

        for epoch in range(num_epochs):
            # print(f'################################ Epoch {epoch+1} START ##########################################')
            # print("\n")
            model.train()
            # model.zero_grad()
            train_loss = 0
           
            #print("++++++++++++++++++ Training Information +++++++++++++++++++++++")
            #print("\n")
            for i,(aa_data, cds_data) in enumerate(tqdm(train_loader)):
                # Get data to cuda if possible
                # wandb.watch(model, cross_entropy_loss, log=None, log_freq=20)

                # aa_data, cds_data = next(iter(train_loader))

                seq_lens = torch.sum(cds_data != -100, dim=1)
                # print("Seq Lens: ", seq_lens)
                seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
                # print("Seq Lens in descending order --->",seq_lens, sorted_index)
                max_seq_len = max(seq_lens)
                # print("Max seq len: ", max_seq_len)

                # sort aa_data and cds_data according to seq_lens in descending order
                # print(type(aa_data))
                # print(type(cds_data))

                aa_data_sorted = []
                cds_data_sorted = []
                for i in range(0, len(sorted_index)):
                    aa_data_sorted.append(aa_data[sorted_index[i]])
                    cds_data_sorted.append(cds_data[sorted_index[i]])
                
                # aa_data = sorted(aa_data, key=lambda x: , reverse=True)
                aa_data_sorted = torch.stack(aa_data_sorted)
                # print("AA DATA", type(aa_data_sorted))
                
                # cds_data = sorted(cds_data, key=lambda x: x.shape[0], reverse=True)
                cds_data_sorted = torch.stack(cds_data_sorted)
                # print("CDS DATA", type(cds_data_sorted))
                

                # print("AA DATA", aa_data_sorted)
                # print("CDS DATA", cds_data_sorted)
                aa_data_sorted = aa_data_sorted.to(rank)
                cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length

                #forward
                output_seq_logits = model(aa_data_sorted, seq_lens, mask=mask)

                """
                Note the output from pad sequence will only contain
                padded sequences till maxm seq length in the current training batch
                So the dimensions error will come if we try to calculate
                loss with output_seq_logits and cds_data.
                Pack pad sequence takes seq length as input and packs and
                when padding packed sequence the pads are added till maxm seq length
                So now the alternative is to remove the extra pads from the cds_data 
                so that it matches the output_seq_logits dimensions
                """
                # print("-----> Output Shape: ",output_seq_logits.shape)
                
                # output_seq_logits_zero_padded = self.get_padded_output(output_seq_logits, seq_lens).to(device)
                # print(output_seq_logits_zero_padded.shape)

            #     # print(output_seq_logits.permute(0,2,1).shape)


                # Trim padding from cds data to match output_seq_logits dimensions 
                # as it is packed padded so containd max len as max seq len in current batch
                cds_pad_trimmed = self.get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) # padding the label
                # print("-----> Target Shape: ",cds_pad_trimmed.shape)
                # print("\n") 

                # print(cds_pad_trimmed[1][:][:])
                # print("\n")
                # print("-----> CDS Pad Trimmed lengths:", cds_pad_trimmed.shape)
                # print(cds_pad_trimmed)
                # print("\n")

                # loss = cross_entropy_loss(output_seq_logits_zero_padded.permute(0,2,1), cds_pad_trimmed.to(device))
                total_loss = cross_entropy_loss(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
                # print("Batch CE Loss: ", loss.item())

                # Get the CAI values for the current batch for predicted and target cds sequences

                # backward
                optimizer.zero_grad()
                total_loss.backward()
                # gradient descent
                optimizer.step()

                # train_batch_count += 1
                train_loss += total_loss.item()

            # print("Train Loss: ", train_loss/len(train_loader))

            avg_train_loss = train_loss / len(train_loader)
            # avg_l1_loss_train = l1_loss_train /  len(train_loader.dataset)
            # avg_l2_loss_train =  l2_loss_train / len(train_loader.dataset)
            # avg_cai_train = total_cai_train / len(train_loader.dataset)
            # avg_stb_train = total_stb_train / len(train_loader.dataset)
            train_losses_epoch.append(avg_train_loss)
            # loss_1_train_epoch.append(avg_l1_loss_train)
            # loss_2_train_epoch.append(avg_l2_loss_train)    
            # accuracy_train = right_tags_train/total_tags_train
            # train_accuracy_epoch.append(accuracy_train)

            
    def test(self, rank, model, test_loader, cross_entropy_loss, cai_loss, stability_loss, cai_type, mask, cds_token_dict, ref_seq_cds):
    
        checkpoint = torch.load('./saved_best_model/best_model-hg19-mfe-lstm.pt', map_location=rank)
        model = RNNModel().to(rank)
        model.load_state_dict(checkpoint['model_state_dict'])
    
        model.eval()
        test_loss = 0
        l1_loss_test = 0
        l2_loss_test = 0
        test_batch_count = 1
        right_tags_test = 0
        total_tags_test = 0
        test_accuracy_epoch = []
        total_cai = 0
        total_stb = 0
        total_gc = 0
        total_cai_original = 0
        cai_original_list = []
        cai_pred_list = []
        total_stb_original = 0
        mfe_original_list = []
        mfe_pred_list = []
        total_gc_original = 0
        gc_original_list = []
        gc_pred_list = []
        predicted_seqs = []
        # print(len(test_loader.dataset))

        with torch.no_grad():
            for i,(aa_data, cds_data) in enumerate(tqdm(test_loader)):
                # print("\n")
                seq_lens = torch.sum(cds_data != -100, dim=1)
                # print("Seq Lens: ", seq_lens)
                seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
                # print("Seq Lens in descending order --->",seq_lens, sorted_index)
                max_seq_len = max(seq_lens)

                aa_data_sorted = []
                cds_data_sorted = []
                for i in range(0, len(sorted_index)):
                    aa_data_sorted.append(aa_data[sorted_index[i]])
                    cds_data_sorted.append(cds_data[sorted_index[i]])
                
                aa_data_sorted = torch.stack(aa_data_sorted)
               
                cds_data_sorted = torch.stack(cds_data_sorted)
        
                aa_data_sorted = aa_data_sorted.to(rank)
                cds_data_sorted = cds_data_sorted.to(rank)

                #forward
                output_seq_logits = model(aa_data_sorted, seq_lens, mask=mask)
                # output_seq_logits_zero_padded = self.get_padded_output(output_seq_logits, seq_lens).to(device)

                # Trim padding from cds data to match output_seq_logits dimensions
                cds_pad_trimmed = self.get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len)


                # loss = cross_entropy_loss(output_seq_logits_zero_padded.permute(0,2,1), cds_pad_trimmed.to(device))
                cel = cross_entropy_loss(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
                # print("Batch CE Loss: ", loss.item())
                if self.cai_type == "softmax":
                    cai_index = cai_loss(output_seq_logits, seq_lens, rank)
                    # print("Batch CAI Loss: ", cai_index.item())
                    total_loss = cel - self.alpha * cai_index
                    
                elif self.cai_type == 'mse' and (self.stability_type=='deg' or self.stability_type=='mfe'):
                    cai_pred, cai_target, predicted_seq = get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, ref_seq_cds, test=True)
                    cai_original_list += list(cai_target)
                    cai_pred_list += list(cai_pred)
                    # predicted_seqs.append(predicted_seq)
                    # print(cai_pred)
                    # cai_required_batch = torch.tensor([self.required_cai]*len(cai_pred)).to(rank)
                    total_cai += torch.sum(cai_pred)
                    total_cai_original += torch.sum(cai_target)
                    # expression_loss = cai_loss(cai_pred, cai_required_batch)
                    

                    gc_pred, gc_target = get_batch_gc_content(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, test=True)
                    gc_original_list += list(gc_target)
                    gc_pred_list += list(gc_pred)
                    total_gc += torch.sum(gc_pred)
                    total_gc_original += torch.sum(gc_target)

                    # stability_pred = get_batch_stability(output_seq_logits, cds_token_dict, seq_lens, self.T, self.pkg, self.stability_type)
                    stability_pred, stability_target = get_batch_stability(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, self.T, self.pkg, self.stability_type, test=True)
                    mfe_original_list += list(stability_target)
                    mfe_pred_list += list(stability_pred)
                    # print(stability_pred)
                    total_stb += torch.sum(stability_pred)
                    total_stb_original += torch.sum(stability_target)
                    total_loss = cel
                    test_loss += total_loss
                    
                    
                    ###### NA ----------
                    # cai_pred, _ = get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, ref_seq_cds)
                    # # print(cai_pred)
                    # cai_required_batch = torch.tensor([self.required_cai]*len(cai_pred)).to(rank)
                    # total_cai += torch.sum(cai_pred)
                    
                    # expression_loss = cai_loss(cai_pred, cai_required_batch)
                    
                    # # stability_pred = get_batch_stability(output_seq_logits, cds_token_dict, seq_lens, self.T, self.pkg, self.stability_type)
                    # stability_pred = get_batch_stability(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, self.T, self.pkg, self.stability_type)
                    # #print(stability_pred)
                    # total_stb += torch.sum(stability_pred)
                    # stability_required_batch = torch.tensor([self.required_stability]*len(stability_pred)).to(rank)
                    # stab_loss = stability_loss(stability_pred, stability_required_batch)
                    # # total_loss = cel + self.alpha * expression_loss + (1-self.alpha) * stab_loss
                    # total_loss = self.bta * cel + self.alpha * expression_loss + (1-self.alpha) * stab_loss
                    # l1_loss_test += expression_loss
                    # l2_loss_test += stab_loss
                    # test_loss += total_loss

                elif self.cai_type == "mse":
                    cai_pred, cai_original = get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, ref_seq_cds)
                    usage_loss = cai_loss(cai_pred, cai_original)
                    # print(f"Batch CAI Loss Validation = {usage_loss} |  Batch CE Loss Validation = {cel}")
                    total_loss = cel + usage_loss
                else:
                    total_loss = cel
                # right_tokens, total_tokens = self.get_correct_tags(output_seq_logits_zero_padded, cds_pad_trimmed, seq_lens)
                right_tokens, total_tokens = self.get_correct_tags(output_seq_logits, cds_pad_trimmed, seq_lens)
                right_tags_test += right_tokens
                total_tags_test += total_tokens
    
        test_accuracy = right_tags_test/total_tags_test
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_l1_loss_test = l1_loss_test / len(test_loader.dataset)
        avg_l2_loss_test = l2_loss_test / len(test_loader.dataset)
        avg_cai = total_cai / len(test_loader.dataset)
        avg_stb = total_stb / len(test_loader.dataset)
        avg_gc = total_gc / len(test_loader.dataset)
        avg_cai_original = total_cai_original / len(test_loader.dataset)
        avg_stb_original = total_stb_original / len(test_loader.dataset)
        avg_gc_original = total_gc_original / len(test_loader.dataset)
        
        print(f'Test CAI = {avg_cai:.4f} | Test CAI Original = {avg_cai_original:.4f} | Test Stability Original = {avg_stb_original:.4f} | Test Stability = {avg_stb:.4f} | Test GC = {avg_gc:.4f} | Test GC Original = {avg_gc_original:.4f} | Test Accuracy = {test_accuracy:.4f} | Test Loss = {avg_test_loss:.4f}')
        # wandb.log({"Test Accuracy": test_accuracy, "Total Test Loss": avg_test_loss, "CAI_avg": avg_cai, "Stability_avg": avg_stb})
        # print(f"Test Accuracy= {test_accuracy:.4f} | Total Test Loss= {avg_test_loss:.4f} | Test Loss CAI= {avg_l1_loss_test:.4f} | CAI Avg = {avg_cai:.4f}  | Test Loss Stab= {avg_l2_loss_test:.4f} | Stability AVg = {avg_stb:.4f} ")

        df = pd.read_csv('./protbert_results/hg19/base_lstm_result/hg19_long_lstm_result.csv')
        df['cai_original'] = [cai.item() for cai in cai_original_list]
        df['cai_pred'] = [cai.item() for cai in cai_pred_list]
        df['mfe_original'] = [mfe.item() for mfe in mfe_original_list]
        df['mfe_pred'] = [mfe.item() for mfe in mfe_pred_list]
        df['gc_original'] = [gc.item() for gc in gc_original_list]
        df['gc_pred'] = [gc.item() for gc in gc_pred_list]
        # df['predicted_cds'] = predicted_seqs
        df.to_csv('./protbert_results/hg19/base_lstm_result/hg19_long_lstm_result.csv', index=False)

    def train(self, rank, mask):
        # setup(rank, world_size=2)
        train_loader, val_loader, test_loader, ref_seq_cds, cds_token_dict  = self.get_token_dict_and_DataLoader()
        print("DATA LOADERS READY")
        print("Length of Test loader dataset: ", len(test_loader.dataset))
        # batch = next(iter(test_loader))
        # print("Batch aa shape: ", batch[0].shape)
        # torch.save(train_loader, './train_loader-stab-mse-ch-10k-150-64.pt')
        # torch.save(val_loader, './val_loader-stab-mse-ch-10k-150-64.pt')
        # torch.save(test_loader, './test_loader-stab-mse-ch-10k-150-64.pt')
        # exit()

        # _, _, _, ref_seq_cds, cds_token_dict = self.get_token_dict_and_DataLoader()
        # train_loader = torch.load('./train_loader-stab-mse-ch-10k-150-64.pt')
        # val_loader = torch.load('./val_loader-stab-mse-ch-10k-150-64.pt')
        # test_loader = torch.load('./test_loader-stab-mse-ch-10k-150-64.pt')
        
        num_epochs = 5

        train_config = {
            "num_epochs": num_epochs,
            "batch_size": 32,
            # "optimizer": "RMSprop",
            "optimizer": "Adam",
            "lr": 0.01,
            "alpha": self.alpha,
            "beta": self.bta,
            "patience": 5,
            "cai_type": self.cai_type,
            "stability_type": self.stability_type,
            "required_cai": self.required_cai,
            "required_stability": self.required_stability,
            "temperature": self.T,
            "tool_pkg": self.pkg,
            "mask": mask,
            "num_layer": 2,
            "num_sequences": len(train_loader.dataset)+len(val_loader.dataset)+len(test_loader.dataset),
        }

        
        # Model
        model = RNNModel()
        # print("MODEL INITIALIZED")
        # if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model, device_ids=[0, 1])
        # torch.cuda.set_device(rank)
        model = model.to(rank)
        # model.cuda(rank)
        # model = DDP(model, device_ids=[rank])

        # model = model.to('cuda')

        # Loss and optimizer
        cross_entropy_loss = CrossEntropyLoss(rank)
        # print("CAI loss dict type", type(cds_token_dict))
        if self.cai_type == "log":
            cai_loss = CAI_LOSS(cds_token_dict=cds_token_dict, ref_seqs=ref_seq_cds).to(rank)
        elif self.cai_type == "softmax":
            cai_loss = SoftmaxCAI(cds_token_dict=cds_token_dict, ref_seqs=ref_seq_cds).to(rank)
        elif self.cai_type == "mse":
            cai_loss = CaiMseLoss(rank)
        
        stability_loss = StabilityMseLoss(rank)

        # cai_loss = SoftmaxCAI(cds_token_dict=cds_token_dict, ref_seqs=ref_seq_cds)
        # print(model.parameters())
        # print(model.bi_gru.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=train_config['lr'])
        # print(model)
        # print("\n")

        # Train Network
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TRAINING STARTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(".")

        # with wandb.init(project="Codon-Optimization-mfe", config = train_config):
        # with wandb.init(project="Codon-Optimization-stab-mse-ch", id = 'e6tjapu9', resume="allow"):
        #   setup(rank, world_size=2)
        # self.train_validate_model(rank, model, train_config, train_loader, val_loader, optimizer, cross_entropy_loss, cai_loss, stability_loss, mask, cds_token_dict, ref_seq_cds)
        #   cleanup()
          # torch.save(model.state_dict(), './model.pt')
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TESTING STARTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.test(rank, model, test_loader, cross_entropy_loss, cai_loss, stability_loss, cai_type, mask, cds_token_dict, ref_seq_cds)
        # self.test(model, test_loader, cross_entropy_loss, cai_loss, stability_loss, cai_type, mask, cds_token_dict, ref_seq_cds)
        # return train_losses_epoch, val_losses_epoch, train_accuracy_epoch, val_accuracy_epoch, test_accuracy


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    # model = RNNModel()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cai_threshold', type=float, help='determines the MSE loss between given threshold CAI and predicted CAI')
    # keep stability threshold between 0-1 if want to use deg(for this optimal value is 0 and worst value is 1) else keep it in -kcal/mol if want to use mfe stability type
    parser.add_argument('--stability_threshold', type=float, help='determines the MSE loss between given threshold stability and predicted stability/degradation')
    parser.add_argument('--cai_type', type=str, help='determines the MSE loss between given threshold CAI and predicted CAI')
    parser.add_argument('--mask', type=str)
    parser.add_argument('--stability_type', type=str, help='mfe or deg , if mfe then mfe from tool else deg from RNAdegformer is taken')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--tool_pkg', type=str)
    parser.add_argument('--temperature', type=int, help='temperature at which mfe is calculated or secondary structure is predicted using tool pkg')
    
    args = parser.parse_args()
    
    dataset_path = './Raw_Data_hg_ecoli_ch/hg19.json'
    cai_type = args.cai_type
    mask = args.mask
    tool_pkg = args.tool_pkg
    required_cai = args.cai_threshold
    required_stability = args.stability_threshold
    stability_type = args.stability_type
    alpha = args.alpha
    temperature = args.temperature
    
    
    
    train_obj = Train(dataset_path, cai_type, required_cai, required_stability, stability_type, alpha, temperature, tool_pkg)

    # train_losses_epoch, val_losses_epoch, train_accuracy_epoch, val_accuracy_epoch, test_accuracy = train_obj.train(cai_type=cai_type, mask=mask)
    
    train_obj.train(rank, mask=mask)
    # mp.spawn(train_obj.train, args=(mask,), nprocs=2)
    # df = pd.DataFrame({
    # 'Epoch': range(1, len(train_losses_epoch)+1),
    # 'Train Loss': train_losses_epoch,
    # 'Validation Loss': val_losses_epoch,
    # 'Train Accuracy': train_accuracy_epoch,
    # 'Validation Accuracy': val_accuracy_epoch,
    # 'Test Accuracy': test_accuracy
    # })

    # df.to_csv('Metrics.csv', index=False)