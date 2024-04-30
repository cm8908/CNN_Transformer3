###################
# Libs
###################

import torch
import torch.nn as nn
import time
from argparse import ArgumentParser

import os

from model_search import TSP_net
from utils import compute_tour_length, AverageMeter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


###################
# Hyper-parameters
###################

parser = ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int, choices=[20, 50, 100], required=True)  # Trained problem size
parser.add_argument('-m', '--segm_len', type=int, choices=[5, 10, 20, 50, 100], default=None)
parser.add_argument('--embedding', type=str, choices=['conv', 'conv_same_padding', 'linear'])
parser.add_argument('--gpu_id', type=str, required=True)

parser.add_argument('--greedy', action='store_true', default=False)
parser.add_argument('--beamsearch', action='store_true', default=False)
args = parser.parse_args()

if args.embedding == 'conv':
    args.nb_neighbors = 10
    args.kernel_size = 11
else:
    args.nb_neighbors = None #20
    args.kernel_size = None

args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
args.nb_layers_decoder = 2
args.nb_heads = 8
args.batchnorm = True  # if batchnorm=True  than batch norm is used
args.max_len_PE = 1000
print(args)

if args.embedding == 'conv':
    embedding_tag = 'cnn' 
elif args.embedding == 'linear':
    embedding_tag = 'lin' 
checkpoint_file = os.path.join('checkpoint', f'tsp{args.nb_nodes}_{embedding_tag}_m{args.segm_len}.pkl')
if not os.path.isfile(checkpoint_file):
    raise Exception(f'File {checkpoint_file} does not exist')

###################
# Hardware : CPU / GPU(s)
###################

device = torch.device("cpu"); gpu_id = -1 # select CPU

gpu_id = args.gpu_id
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
###################
# Instantiate a training network and a baseline network
###################
try: 
    del model_baseline # remove existing model
except:
    pass

model_baseline = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size, 
                         args.dim_input_nodes, args.dim_emb, args.dim_ff, 
                         args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,  args.max_len_PE,
                         segm_len=args.segm_len, batchnorm=args.batchnorm)

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


greedy = args.greedy; beamsearch = args.beamsearch 
if greedy and not beamsearch:
    B = 1




###################   
# Test set
###################

for test_size in [20, 50, 100]:

    ###################   
    # Test set
    ###################
    if test_size == 20:
        x_10k = torch.load('data/10k_TSP20.pt').to(device)
        x_10k_len = torch.load('data/10k_TSP20_len.pt').to(device)
        L_concorde = x_10k_len.mean().item()
        args.bsz = 100
        args.nb_batch_eval = 10_000 // args.bsz
        B = 420
    if test_size == 50:
        x_10k = torch.load('data/10k_TSP50.pt').to(device)
        x_10k_len = torch.load('data/10k_TSP50_len.pt').to(device)
        L_concorde = x_10k_len.mean().item()
        args.bsz = 10
        args.nb_batch_eval = 10_000 // args.bsz
        B = 2500
    if test_size == 100:
        x_10k = torch.load('data/10k_TSP100.pt').to(device)
        x_10k_len = torch.load('data/10k_TSP100_len.pt').to(device)
        L_concorde = x_10k_len.mean().item()
        args.bsz = 5
        args.nb_batch_eval = 10_000 // args.bsz
        B = 2500
    if test_size == 200:
        raise NotImplementedError()
        x_10k = torch.load('data/10k_TSP200.pt').to(device)
        x_10k_len = torch.load('data/10k_TSP200_len.pt').to(device)
        L_concorde = x_10k_len.mean().item()
        args.bsz = 2
        args.nb_batch_eval = 1_000 // args.bsz
        B = 2500
    assert 10_000 % args.bsz == 0
    nb_TSPs = args.nb_batch_eval* args.bsz

    ###################   
    # Run beam search
    ###################
    start = time.time()
    mean_tour_length_greedy = AverageMeter()
    mean_tour_length_beamsearch = AverageMeter()
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
                x_len_greedy = L_greedy
                gap_greedy.update((x_len_greedy/ x_len_concorde - 1.0).mean().item())
            # beamsearch
            if beamsearch:
                tours_beamsearch = tours_beamsearch.view(args.bsz*B, test_size)
                x = x.repeat_interleave(B,dim=0)
                L_beamsearch = compute_tour_length(x, tours_beamsearch)
                # L_beamsearch = compute_distance(x_org, tours_greedy, metric)
                tours_beamsearch = tours_beamsearch.view(args.bsz, B, test_size)
                L_beamsearch = L_beamsearch.view(args.bsz, B)
                L_beamsearch_tmp = L_beamsearch
                L_beamsearch, idx_min = L_beamsearch.min(dim=1)
                mean_tour_length_beamsearch.update(L_beamsearch.mean().item())
                x_len_beamsearch = L_beamsearch
                gap_beamsearch.update((x_len_beamsearch/ x_len_concorde - 1.0).mean().item())
        torch.cuda.empty_cache() # free GPU reserved memory 
    tot_time = time.time()-start
    

    
    ###################   
    # Write result file
    ###################
    
    csv_file = 'test_varsize_results.csv'
    if not os.path.isfile(csv_file):
        print(f'File {csv_file} not found. Writing new file...')
        with open(csv_file, 'a') as f:
            head_str = ','.join(['segm_len', 'train_size', 'test_size', 'L_greedy', 'gap_greedy', 'L_beamsearch', 'gap_beamsearch']) + '\n'
            f.write(head_str)
    metrics_str = list(map(str, [args.segm_len, args.nb_nodes, test_size, 
                                 mean_tour_length_greedy.avg, 100*gap_greedy.avg,
                                 mean_tour_length_beamsearch.avg, 100*gap_beamsearch.avg]))
    metrics_str = ','.join(metrics_str) + '\n'
    with open(csv_file, 'a') as f:
        f.write(metrics_str)
    

    nb_TSPs = args.nb_batch_eval* args.bsz
    if not os.path.isdir('test_res'):
        os.mkdir('test_res')
    file_name = os.path.join('test_res', 'variable_size.txt')
    with open(file_name, 'a', 1) as file:
        mystring = f'trained on {args.nb_nodes}, tested on {test_size}, nb_TSPs: {nb_TSPs}, B: {B}, L_greedy: {mean_tour_length_greedy.avg}, gap_greedy(%): {100*gap_greedy.avg}, L_beamsearch: {mean_tour_length_beamsearch.avg}, gap_beamsearch(%): {100*gap_beamsearch.avg}, L_concorde: {L_concorde}\n'
        print(mystring)
        file.write(mystring)


