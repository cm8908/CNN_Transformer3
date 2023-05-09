# CNN_Transformer
Code for paper: [A Lightweight CNN-Transformer Model for Learning Traveling Salesman Problems](https://arxiv.org/abs/2305.01883)

# Runs (Examples)
To train the proposed model on TSP20, run the following:
```console
$ train.py --exp_name Experiment-Name -n 20 --bsz 512 --gpu_id 0 --embedding conv --nb_neighbors 10 --kernel_size 11 --segm_len 5 --batchnorm
```
To test the trained model on TSP20 random dataset, run the following:
```console
$ test.py -n 20 --embedding conv --nb_neighbors 10 --kernel_size 11 --gpu_id 0 --ckpt_file Experiment-Name-on-Train.pkl --exp_tag Result-File-Name --segm_len 5 --greedy --beamsearch --batchnorm
```
OR you can just run shell scripts for train and test respectively. You may modify the .sh files uploaded on this repository if necessary.
```console
$ sh train_sh/train-sh-filename.sh
$ sh test_sh/test-sh-filename.sh
```
You can use `test_tsplib.ipynb` to test the trained model on the TSPLIB dataset

The pretrained model checkpoints are uploaded in the `checkpoint` directory
# Requirements
- pytorch 1.12.1
- numpy 1.22.3
- pandas 1.4.3
- python 3.10.5
- matplotlib 3.7.1

# Acknowledgement
The codes here are based on the work of Xavier Bresson's [TSP Transformer](https://github.com/xbresson/TSP_Transformer)
