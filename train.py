

###################
# Libs
###################

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from argparse import ArgumentParser

import os
import datetime

from model import TSP_net
from utils import compute_tour_length
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


###################
# Hyper-parameters
###################
embedding_choices = ['linear', 'conv_same_padding', 'conv', 'conv_linear']  # conv_XY, conv_same_padding, conv_linear
parser = ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--exp_detail', type=str, default='')
parser.add_argument('-n', '--nb_nodes', type=int, choices=[20, 50, 100, 150, 200], required=True)
parser.add_argument('--bsz', type=int, required=True)
parser.add_argument('--gpu_id', type=str, required=True)

parser.add_argument('--embedding', type=str, choices=embedding_choices, required=True)
parser.add_argument('--nb_neighbors', type=int, default=None)  # For CNN
parser.add_argument('--kernel_size', type=int, default=None)  # For CNN

parser.add_argument('--segm_len', type=int, default=None)  
parser.add_argument('--batchnorm', action='store_true', default=False)

parser.add_argument('--resume_training', action='store_true', default=False)
parser.add_argument('--resume_file', type=str, default=None)

parser.add_argument('--dim_emb', type=int, default=128)
parser.add_argument('--dim_ff', type=int, default=512)
parser.add_argument('--dim_input_nodes', type=int, default=2)
parser.add_argument('--nb_layers_encoder', type=int, default=6)
parser.add_argument('--nb_layers_decoder', type=int, default=2)
parser.add_argument('--nb_heads', type=int, default=8)
parser.add_argument('--nb_epochs', type=int, default=100)
parser.add_argument('--nb_batch_per_epoch', type=int, default=2500)
parser.add_argument('--nb_batch_eval', type=int, default=20)
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument('--tol', type=int, default=1e-3)
parser.add_argument('--max_len_PE', type=int, default=1000)

parser.add_argument('--fp16', action='store_true', default=False)
args = parser.parse_args()
if args.fp16:
    raise NotImplementedError('fp16 not implemented')
if args.embedding == 'conv':
    assert not args.nb_neighbors is None
    assert not args.kernel_size is None
    assert args.nb_neighbors == args.kernel_size - 1
if args.resume_training:
    assert not args.resume_file is None

print(args)
if not args.debug: 
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    writer = SummaryWriter('./logs/'+args.exp_name)
else:
    print("########### RUNNING ON DEBUG MODE ############")


###################
# Hardware : CPU / GPU(s)
###################

device = torch.device("cpu"); gpu_id = -1 # select CPU

gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
# if args.debug:
#     device = torch.device("cpu"); gpu_id = -1 # select CPU
print(device)


###################
# Small test set for quick algorithm comparison
# Note : this can be removed
###################

save_1000tsp = True
save_1000tsp = False
if save_1000tsp:
    bsz = 1000
    x = torch.rand(bsz, args.nb_nodes, args.dim_input_nodes, device='cpu') 
    print(x.size(),x[0])
    data_dir = os.path.join("data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if args.nb_nodes==20 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp20"))
    if args.nb_nodes==50 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp50"))
    if args.nb_nodes==100 : torch.save({ 'x': x, }, '{}.pkl'.format(data_dir + "/1000tsp100"))

checkpoint = None
if args.nb_nodes==20 : checkpoint = torch.load("data/1000tsp20.pkl")
if args.nb_nodes==50 : checkpoint = torch.load("data/1000tsp50.pkl")
if args.nb_nodes==100 : checkpoint = torch.load("data/1000tsp100.pkl")
if checkpoint is not None:
    x_1000tsp = checkpoint['x'].to(device)
    n = x_1000tsp.size(1)
    print('nb of nodes :',n)
else:
    x_1000tsp = torch.rand(1000, args.nb_nodes, args.dim_input_nodes, device=device)
    n = x_1000tsp.size(1)
    print('nb of nodes :',n)

    
###################
# Instantiate a training network and a baseline network
###################
try: 
    del model_train # remove existing model
    del model_baseline # remove existing model
except:
    pass

model_train = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size,
                      args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder, 
                      args.nb_heads, args.max_len_PE, args.segm_len, batchnorm=args.batchnorm)

model_baseline = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size,
                      args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder, 
                      args.nb_heads, args.max_len_PE, args.segm_len, batchnorm=args.batchnorm)

# uncomment these lines if trained with multiple GPUs
print(torch.cuda.device_count())
if torch.cuda.device_count()>1:
    model_train = nn.DataParallel(model_train)
    model_baseline = nn.DataParallel(model_baseline)
# uncomment these lines if trained with multiple GPUs

if args.fp16:
    scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
model_baseline.eval()

print(args); print('')

# Logs
if not args.debug:
    time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    # file_name = 'logs'+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
    file_name = f'logs/{args.exp_name}.txt'
    file = open(file_name,"w",1) 
    file.write(time_stamp+'\n\n') 
    file.write(args.exp_detail+'\n\n')
    for arg in vars(args):
        file.write(arg)
        hyper_param_val="={}".format(getattr(args, arg))
        file.write(hyper_param_val)
        file.write('\n')
    file.write('\n\n') 
else:
    torch.autograd.set_detect_anomaly(True)
plot_performance_train = []
plot_performance_baseline = []
epoch_ckpt = 0
tot_time_ckpt = 0


# Uncomment these lines to re-start training with saved checkpoint
checkpoint_dir = 'checkpoint'
if args.resume_training:
    checkpoint_file = os.path.join(checkpoint_dir, args.resume_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    epoch_ckpt = checkpoint['epoch'] + 1
    tot_time_ckpt = checkpoint['tot_time']
    plot_performance_train = checkpoint['plot_performance_train']
    plot_performance_baseline = checkpoint['plot_performance_baseline']
    model_baseline.load_state_dict(checkpoint['model_baseline'])
    model_train.load_state_dict(checkpoint['model_train'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
    del checkpoint
# Uncomment these lines to re-start training with saved checkpoint

###################
# Main training loop 
###################
start_training_time = time.time()

for epoch in tqdm(range(0,args.nb_epochs), 'Training'):
    
    # re-start training with saved checkpoint
    epoch += epoch_ckpt
    if epoch == args.nb_epochs:
        break

    ###################
    # Train model for one epoch
    ###################
    start = time.time()
    model_train.train() 

    for step in range(1,args.nb_batch_per_epoch+1):    

        # generate a batch of random TSP instances    
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 
            
        # compute tours for model
        with torch.cuda.amp.autocast(enabled=args.fp16):
            tour_train, sumLogProbOfActions = model_train(x, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
      
            # compute tours for baseline
            with torch.no_grad():
                tour_baseline, _ = model_baseline(x, deterministic=True)

        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train) # size(L_train)=(bsz)
        L_baseline = compute_tour_length(x, tour_baseline) # size(L_baseline)=(bsz)
        
        # backprop
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

        
    ###################
    # Evaluate train model and baseline on 10k random TSP instances
    ###################
    model_train.eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for step in range(0,args.nb_batch_eval):

        # generate a batch of random tsp instances   
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            # compute tour for model and baseline
            with torch.no_grad():
                tour_train, _ = model_train(x, deterministic=True)
                tour_baseline, _ = model_baseline(x, deterministic=True)
            
            # get the lengths of the tours
            L_train = compute_tour_length(x, tour_train)
            L_baseline = compute_tour_length(x, tour_baseline)

            # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
            mean_tour_length_train += L_train.mean().item()
            mean_tour_length_baseline += L_baseline.mean().item()

    mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
    mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval

    # evaluate train model and baseline and update if train model is better
    update_baseline = mean_tour_length_train+args.tol < mean_tour_length_baseline
    if update_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )

    # Compute TSPs for small test set
    # Note : this can be removed
    with torch.cuda.amp.autocast(enabled=args.fp16):
        with torch.no_grad():
            tour_baseline, _ = model_baseline(x_1000tsp, deterministic=True)
        mean_tour_length_test = compute_tour_length(x_1000tsp, tour_baseline).mean().item()
    
    # For checkpoint
    plot_performance_train.append([ (epoch+1), mean_tour_length_train])
    plot_performance_baseline.append([ (epoch+1), mean_tour_length_baseline])
        
    # Compute optimality gap
    if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
    elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
    else: gap_train = -1.0
    
    # Print and save in txt file
    mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_train: {:.3f}, L_base: {:.3f}, L_test: {:.3f}, gap_train(%): {:.3f}, update: {}'.format(
        epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_train, mean_tour_length_baseline, mean_tour_length_test, 100*gap_train, update_baseline) 
    print(mystring_min) # Comment if plot display
    if not args.debug:
        writer.add_scalar('mean_tour_len_train', mean_tour_length_train, epoch)
        writer.add_scalar('mean_tour_len_baseline', mean_tour_length_baseline, epoch)
        
        file.write(mystring_min+'\n')

    
        # Saving checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filename = f'{checkpoint_dir}/{args.exp_name}.pkl'
        if (epoch+1) % 10 == 0:
            filename = f'{checkpoint_dir}/{args.exp_name}_e{epoch}.pkl'
        torch.save({
            'epoch': epoch,
            'time': time_one_epoch,
            'tot_time': time_tot,
            'loss': loss.item(),
            'TSP_length': [torch.mean(L_train).item(), torch.mean(L_baseline).item(), mean_tour_length_test],
            'plot_performance_train': plot_performance_train,
            'plot_performance_baseline': plot_performance_baseline,
            'mean_tour_length_test': mean_tour_length_test,
            'model_baseline': model_baseline.state_dict(),
            'model_train': model_train.state_dict(),
            'whole_model': model_baseline,
            'optimizer': optimizer.state_dict(),
            }, filename)

if not args.debug:
    writer.close()
      
