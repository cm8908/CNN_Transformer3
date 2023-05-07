# CNN_Transformer
Code for paper: [A Lightweight CNN-Transformer Model for Learning Traveling Salesman Problems](https://arxiv.org/abs/2305.01883)

# Runs (Examples)
To train the proposed model on TSP20, run the following:
`$ train.py --exp_name Experiment-Name -n 20 --bsz 512 --gpu_id 0 --embedding conv --nb_neighbors 10 --kernel_size 11 --segm_len 5 --batchnorm`


# Requirements
- pytorch 1.12.1
- numpy 1.22.3
- pandas 1.4.3
- 3.10.5
