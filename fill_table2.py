import os
import torch
import time
import nvidia_smi
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def get_model(args, is_train=False):
    if is_train:
        from model import TSP_net
    else:
        from model_search import TSP_net
    embedding = args.embedding
    nb_neighbors = 10
    kernel_size = nb_neighbors + 1
    segm_len = args.segm_len
    batchnorm = True
    return TSP_net(embedding, nb_neighbors, kernel_size, dim_input_nodes=2, dim_emb=128, dim_ff=512,
                   nb_layers_encoder=6, nb_layers_decoder=2, nb_heads=8, max_len_PE=1000, segm_len=segm_len, batchnorm=batchnorm)
def main(args):
    result_dict = {}
    
    # Setup Experiment tag
    embedding_tag = 'cnn' if args.embedding == 'conv' else 'lin'
    exp_tag = f'tsp{args.nb_nodes}_{embedding_tag}_m{args.segm_len}.pkl'
    result_dict['Experiment'] = exp_tag
    print('Experiment:', exp_tag)
    
    # Setup Device and Get 10k Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_10k = torch.load(f'data/10k_TSP{args.nb_nodes}.pt').to(device)
    print('Device:', device, args.gpu_id)

    # Get Train Model and Measure T-Time
    model = get_model(args, is_train=True).to(device)
    t_time_s = time.time()
    if args.bsz is None:
        model(data_10k, deterministic=False)
    else:
        assert 10_000 % args.bsz == 0
        for i in tqdm(range(0, 10_000, args.bsz)):
            minibatch = data_10k[i:i+args.bsz]
            model(minibatch, deterministic=False)
    t_time = time.time() - t_time_s
    result_dict['T-Time (total)'] = t_time
    print('T-Time (total):', t_time)
    t_time = t_time / 10_000 if args.bsz is not None else t_time
    result_dict['T-Time (per instance)'] = t_time
    print('T-Time (per instance):', t_time)

    # Measure Memory Usage
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    result_dict['Memory Usage'] = info.used
    print('Memory Usage:', info.used)
    nvidia_smi.nvmlShutdown()
    torch.cuda.empty_cache()

    # Measure I Time Greedy
    model_infer = get_model(args, is_train=False).to(device)
    if args.bsz is not None:
        i_time_s = time.time()
        for i in tqdm(range(0, args.nb_instances_eval, args.bsz)):
            minibatch = data_10k[i:i+args.bsz]
            model_infer(minibatch, greedy=True, beamsearch=False, B=1)
        i_time_g = time.time() - i_time_s
        result_dict['I-Time Greedy (total)'] = i_time_g
        print('I-Time Greedy (total):', i_time_g)
        i_time_g /= args.nb_instances_eval
        result_dict['I-Time Greedy (per instance)'] = i_time_g
        print('I-Time Greedy (per instance):', i_time_g)
        torch.cuda.empty_cache()
    
    # Measure I Time Beamsearch
    if args.bsz is not None:
        i_time_s = time.time()
        for i in tqdm(range(0, args.nb_instances_eval, args.bsz)):
            minibatch = data_10k[i:i+args.bsz]
            model_infer(minibatch, greedy=False, beamsearch=True, B=2500)
        i_time_bs = time.time() - i_time_s
        result_dict['I-Time Beamsearch (total)'] = i_time_bs
        print('I-Time Beamsearch (total):', i_time_bs)
        i_time_bs /= args.nb_instances_eval
        result_dict['I-Time Beamsearch (per instance)'] = i_time_bs
        print('I-Time Beamsearch (per instance):', i_time_bs)
    
    # Write to csv
    csv_file = 'table2_results.csv'
    if not os.path.isfile(csv_file):
        print(f'File {csv_file} not found. Writing new file...')
        with open(csv_file, 'a') as f:
            head_str = ','.join(result_dict.keys()) + '\n'
            f.write(head_str)
    metrics_str = list(map(str, result_dict.values()))
    metrics_str = ','.join(metrics_str) + '\n'
    with open(csv_file, 'a') as f:
        f.write(metrics_str)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nb_nodes', type=int, choices=[20, 50, 100], required=True)
    parser.add_argument('-m', '--segm_len', type=int, default=None)
    parser.add_argument('--embedding', type=str, choices=['conv', 'linear'], required=True)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--bsz', type=int, default=None)
    parser.add_argument('--nb_instances_eval', type=int, default=10_000)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    main(args)
