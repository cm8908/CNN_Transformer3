###################
# Libs
###################

import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

from model_search import TSP_net
from utils import AverageMeter, compute_tour_length, csv_write
# visualization 
from IPython.display import set_matplotlib_formats, clear_output
set_matplotlib_formats('png2x','pdf')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: 
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    from concorde.tsp import TSPSolver # !pip install -e pyconcorde
except:
    pass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


###################
# Hyper-parameters
###################

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int, choices=[20, 50, 100, 150, 200], default=20)#required=True)
parser.add_argument('--embedding', type=str, choices=['linear', 'conv_same_padding', 'conv', 'conv2'], required=True)
parser.add_argument('--nb_neighbors', type=int, default=None)
parser.add_argument('--kernel_size', type=int, default=None)
parser.add_argument('--gpu_id', type=str, required=True)
parser.add_argument('--ckpt_file', type=str, required=True)
parser.add_argument('--exp_tag', type=str, required=True)
parser.add_argument('--segm_len', type=int, default=None)

parser.add_argument('--beam_width', type=int, default=2500)
parser.add_argument('--greedy', action='store_true', default=False)
parser.add_argument('--beamsearch', action='store_true', default=False)

parser.add_argument('--batchnorm', action='store_true', default=False)
parser.add_argument('--nb_layers_decoder', type=int, default=2)
args = parser.parse_args()

args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
# args.nb_layers_decoder = 2
args.nb_heads = 8
# args.batchnorm = True  # if batchnorm=True  than batch norm is used
args.max_len_PE = 1000
print(args)

###################
# Hardware : CPU / GPU(s)
###################

device = torch.device("cpu"); gpu_id = -1 # select CPU

gpu_id = args.gpu_id # select a single GPU  
# gpu_id = '2,3' # select multiple GPUs  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
print(device)


###################
# Network definition
# Notation : 
#            bsz : batch size
#            nb_nodes : number of nodes/cities
#            dim_emb : embedding/hidden dimension
#            nb_heads : nb of attention heads
#            dim_ff : feed-forward dimension
#            nb_layers : number of encoder/decoder layers
###################

# # Instantiate, Load Model

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
###################
# Instantiate a training network and a baseline network
###################
try: 
    del model_baseline # remove existing model
except:
    pass

model_baseline = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size, 
                      args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder, 
                      args.nb_heads, args.max_len_PE, args.segm_len,
                      batchnorm=args.batchnorm)

# uncomment these lines if trained with multiple GPUs
print(torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model_baseline = nn.DataParallel(model_baseline)
# uncomment these lines if trained with multiple GPUs

model_baseline = model_baseline.to(device)
model_baseline.eval()

print(args); print('')



###################
# Load checkpoint
###################
# checkpoint_file = "checkpoint/20_conv_k-10_sort-byy_desc_n20_gpu2_conv_k10_22-11-07--11-55-56.pkl"
checkpoint_file = 'checkpoint/'+args.ckpt_file
checkpoint = torch.load(checkpoint_file, map_location=device)
epoch_ckpt = checkpoint['epoch'] + 1
tot_time_ckpt = checkpoint['tot_time']
plot_performance_train = checkpoint['plot_performance_train']
plot_performance_baseline = checkpoint['plot_performance_baseline']
model_baseline.load_state_dict(checkpoint['model_baseline'])
print('Load checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
del checkpoint

mystring_min = 'Epoch: {:d}, tot_time_ckpt: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}\n'.format(
    epoch_ckpt, tot_time_ckpt/3660/24, plot_performance_train[-1][1], plot_performance_baseline[-1][1]) 
print(mystring_min) 

###################   
# Hyper-parameter for beam search
###################
B = args.beam_width; greedy = args.greedy; beamsearch = args.beamsearch 


# # Prepare Data and Run Beamsearch

###################   
# Test set
###################
if args.nb_nodes == 20:
    x_10k = torch.load('data/10k_TSP20.pt').to(device)
    x_10k_len = torch.load('data/10k_TSP20_len.pt').to(device)
    L_concorde = x_10k_len.mean().item()
    args.bsz = 100
    args.nb_batch_eval = 10_000 // args.bsz
    # B = 420
if args.nb_nodes == 50:
    x_10k = torch.load('data/10k_TSP50.pt').to(device)
    x_10k_len = torch.load('data/10k_TSP50_len.pt').to(device)
    L_concorde = x_10k_len.mean().item()
    args.bsz = 10
    args.nb_batch_eval = 10_000 // args.bsz
    # B = 2500
if args.nb_nodes == 100:
    x_10k = torch.load('data/10k_TSP100.pt').to(device)
    x_10k_len = torch.load('data/10k_TSP100_len.pt').to(device)
    L_concorde = x_10k_len.mean().item()
    args.bsz = 5
    args.nb_batch_eval = 10_000 // args.bsz
    # B = 2500
if args.nb_nodes == 200:
    # raise NotImplementedError()
    x_10k = torch.load('data/10k_TSP200.pt').to(device)
    x_10k_len = torch.load('data/10k_TSP200_len.pt').to(device)
    L_concorde = x_10k_len.mean().item()
    args.bsz = 10
    args.nb_batch_eval = 10_000 // args.bsz
    # B = 2500
assert 10_000 % args.bsz == 0
nb_TSPs = args.nb_batch_eval* args.bsz


file_name = f"test_res/{args.exp_tag}.txt"
file = open(file_name,"a",1) 
file.write('\n'.join([f'{k}:{v}' for k, v in vars(args).items()])+'\n\n')
file.write(mystring_min+'\n')

###################   
# Run beam search
###################
start = time.time()
mean_tour_length_greedy = AverageMeter()
mean_tour_length_beamsearch = AverageMeter()
mean_scores_greedy = AverageMeter()
mean_scores_beamsearch = AverageMeter()
gap_greedy = AverageMeter()
gap_beamsearch = AverageMeter()
for step in range(0,args.nb_batch_eval):
    print('batch index: {}, tot_time: {:.3f}min'.format(step, (time.time()-start)/60))
    # extract a batch of test tsp instances 
    x = x_10k[step*args.bsz:(step+1)*args.bsz,:,:]
    # x = torch.rand(args.bsz, test_size, 2, device=device)
    x_len_concorde = x_10k_len[step*args.bsz:(step+1)*args.bsz]
    # compute tour for model and baseline
    with torch.no_grad():
        tours_greedy, tours_beamsearch, scores_greedy, scores_beamsearch = model_baseline(x, B, greedy, beamsearch)
        # greedy
        if greedy:
            L_greedy = compute_tour_length(x, tours_greedy)
            # L_greedy = compute_distance(x_org, tours_greedy, metric)
            mean_tour_length_greedy.update(L_greedy.mean().item())
            mean_scores_greedy.update(scores_greedy.mean().item())
            x_len_greedy = L_greedy
            gap_greedy.update((x_len_greedy/ x_len_concorde - 1.0).mean())
        # beamsearch
        if beamsearch:
            tours_beamsearch = tours_beamsearch.view(args.bsz*B, args.nb_nodes)
            x = x.repeat_interleave(B,dim=0)
            L_beamsearch = compute_tour_length(x, tours_beamsearch)
            # L_beamsearch = compute_distance(x_org, tours_greedy, metric)
            tours_beamsearch = tours_beamsearch.view(args.bsz, B, args.nb_nodes)
            L_beamsearch = L_beamsearch.view(args.bsz, B)
            L_beamsearch_tmp = L_beamsearch
            L_beamsearch, idx_min = L_beamsearch.min(dim=1)
            mean_tour_length_beamsearch.update(L_beamsearch.mean().item())
            mean_scores_beamsearch.update(scores_beamsearch.mean().item())
            x_len_beamsearch = L_beamsearch
            gap_beamsearch.update((x_len_beamsearch/ x_len_concorde - 1.0).mean())
    torch.cuda.empty_cache() # free GPU reserved memory 
    if step % 100 == 0:
        record_str = f'Step:{step} - L_greedy: {mean_tour_length_greedy.avg}, gap_greedy: {gap_greedy.avg}, L_beamsearch: {mean_tour_length_beamsearch.avg}, gap_beamsearch: {gap_beamsearch.avg}\n'
        file.write(record_str)

tot_time = time.time()-start
    

    
###################   
# Write result file
###################
nb_TSPs = args.nb_batch_eval* args.bsz
# file_name = f"test_res/{args.exp_tag}.txt"
# file = open(file_name,"a",1) 
mystring = '\Tag:{}, Embedding:{} --- nb_nodes: {:d}, nb_TSPs: {:d}, B: {:d}, L_greedy: {:.6f}, L_concorde: {:.5f}, L_beamsearch: {:.5f}, \
gap_greedy(%): {:.5f}, gap_beamsearch(%): {:.5f}, scores_greedy: {:.5f}, scores_beamsearch: {:.5f}, tot_time: {:.4f}min, \
tot_time: {:.3f}hr, mean_time: {:.3f}sec'.format(args.exp_tag, args.embedding, args.nb_nodes, nb_TSPs, B, mean_tour_length_greedy.avg, L_concorde, \
                                mean_tour_length_beamsearch.avg, 100*gap_greedy.avg, 100*gap_beamsearch.avg, mean_scores_greedy.avg, \
                                mean_scores_beamsearch.avg, tot_time/60, tot_time/3600, tot_time/nb_TSPs)
print(mystring)
file.write(mystring)
csv_write(args.exp_tag, mean_tour_length_greedy.avg, 100*gap_greedy.avg, mean_tour_length_beamsearch.avg, 100*gap_beamsearch.avg, tot_time/nb_TSPs)
file.close()


