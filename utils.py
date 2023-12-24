import torch
import os
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0
def compute_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, nb_nodes, 2) batch of tsp tour instances
             tour of size (bsz, nb_nodes) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    first_cities = x[range(bsz), tour[:,0], :] # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1,nb_nodes):
            current_cities = x[range(bsz), tour[:,i]] 
            L += (current_cities - previous_cities).pow(2).sum(dim=1).sqrt()
            # L += torch.sum( torch.round((current_cities - previous_cities)**2) , dim=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += (current_cities - first_cities).pow(2).sum(dim=1).sqrt()
        # L += torch.sum( torch.round((current_cities - first_cities)**2) , dim=1 )**0.5 # dist(last, first node)  
    return L

def csv_write(tag, L_greedy, gap_greedy, L_beamsearch, gap_beamsearch, mean_time, csv_file='./test_results.csv'):
    if not os.path.isfile(csv_file):
        print(f'File {csv_file} not found. Writing new file...')
        with open(csv_file, 'a') as f:
            head_str = ','.join(['tag', 'L_greedy', 'gap_greedy', 'L_beamsearch', 'gap_beamsearch', 'mean_time']) + '\n'
            f.write(head_str)
    args_str = list(map(str, [tag, L_greedy, gap_greedy.item(), L_beamsearch, gap_beamsearch.item(), mean_time]))
    args_str = ','.join(args_str) + '\n'
    with open(csv_file, 'a') as f:
        f.write(args_str)