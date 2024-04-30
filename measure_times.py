import os
import torch
from timeit import timeit
from tqdm import tqdm
from argparse import ArgumentParser
from utils import compute_tour_length

def get_model(args, is_train=False):
    if is_train:
        from model import TSP_net
    else:
        from model_search import TSP_net
    embedding = args.embedding
    if embedding == 'conv':
        nb_neighbors = 10
        kernel_size = nb_neighbors + 1
    else:
        nb_neighbors = None
        kernel_size = None
    segm_len = args.segm_len
    batchnorm = True
    return TSP_net(embedding, nb_neighbors, kernel_size, dim_input_nodes=2, dim_emb=128, dim_ff=512,
                   nb_layers_encoder=6, nb_layers_decoder=2, nb_heads=8, max_len_PE=1000, segm_len=segm_len, batchnorm=batchnorm)

def forward_backward_pass(model, model_baseline, data, optimizer, args, nb_batch):
    for i in tqdm(range(nb_batch)):
        minibatch = data[i*args.train_bsz:(i+1)*args.train_bsz]
        tour, slp = model(minibatch, deterministic=False)
        with torch.no_grad():
            tour_b, _ = model_baseline(minibatch, deterministic=False)
        L_train, L_baseline = compute_tour_length(minibatch, tour), compute_tour_length(minibatch, tour_b)
        loss = ((L_train - L_baseline) * slp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def forward_greedy(model, data, args, nb_batch):
    for i in tqdm(range(nb_batch)):
        minibatch = data[i*args.greedy_bsz:(i+1)*args.greedy_bsz]
        model(minibatch, greedy=True, beamsearch=False, B=1)
def forward_beamsearch(model, data, args, nb_batch):
    for i in tqdm(range(nb_batch)):
        minibatch = data[i*args.beamsearch_bsz:(i+1)*args.beamsearch_bsz]
        model(minibatch, greedy=False, beamsearch=True, B=2500)

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
    model_baseline = get_model(args, is_train=True).to(device)
    model.train()
    model_baseline.eval()
    oz = torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.train_bsz is not None:
        print('> Train')
        nb_batch_train = 10_000 // args.train_bsz
        if not 10_000 % args.train_bsz == 0:
            nb_batch_train += 1 
        t_time = timeit(lambda: forward_backward_pass(model, model_baseline, data_10k, oz, args, nb_batch_train), number=args.nb_iter)
        t_time /= args.nb_iter
        result_dict['T-Time (total)'] = t_time
        print('T-Time (total):', t_time)
        t_time = t_time / 10_000 if args.train_bsz is not None else t_time
        result_dict['T-Time (per instance)'] = t_time
        print('T-Time (per instance):', t_time)
        torch.cuda.empty_cache()

    # Measure Memory Usage
    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.gpu_id)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # result_dict['Memory Usage'] = info.used
    # print('Memory Usage:', info.used)
    # nvidia_smi.nvmlShutdown()
    # torch.cuda.empty_cache()

    # Measure I Time Greedy
    model_infer = get_model(args, is_train=False).to(device)
    model_infer.eval()
    with torch.no_grad():
        if args.greedy_bsz is not None:
            print('> Greedy')
            nb_batch_eval_greedy = args.nb_instances_eval // args.greedy_bsz
            if not args.nb_instances_eval % args.greedy_bsz == 0:
                nb_batch_eval_greedy += 1
            i_time_g = timeit(lambda: forward_greedy(model_infer, data_10k, args, nb_batch_eval_greedy), number=args.nb_iter)
            i_time_g /= args.nb_iter
            result_dict['I-Time Greedy (total)'] = i_time_g
            print('I-Time Greedy (total):', i_time_g)
            # i_time_g /= args.nb_instances_eval
            i_time_g /= args.nb_instances_eval
            result_dict['I-Time Greedy (per instance)'] = i_time_g
            print('I-Time Greedy (per instance):', i_time_g)
            torch.cuda.empty_cache()
    
        # Measure I Time Beamsearch
        if args.beamsearch_bsz is not None:
            print('> Beamsearch')
            nb_batch_eval_beamsearch = args.nb_instances_eval // args.beamsearch_bsz
            if not args.nb_instances_eval % args.beamsearch_bsz == 0:
                nb_batch_eval_beamsearch += 1
            i_time_bs = timeit(lambda: forward_beamsearch(model_infer, data_10k, args, nb_batch_eval_beamsearch), number=args.nb_iter)
            i_time_bs /= args.nb_iter
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
    parser.add_argument('--train_bsz', type=int, default=None)
    parser.add_argument('--greedy_bsz', type=int, default=None)
    parser.add_argument('--beamsearch_bsz', type=int, default=None)
    parser.add_argument('--nb_instances_eval', type=int, default=10_000)
    parser.add_argument('--nb_iter', default=10, type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    main(args)
