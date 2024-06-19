# CNN_Transformer
Code for paper: [A Lightweight CNN-Transformer Model for Learning Traveling Salesman Problems](https://arxiv.org/abs/2305.01883)

# Runs (Examples)
(Updated 04-30-2024) Now you can use [Docker](https://hub.docker.com/r/fmsjung/lit-cnn-trnsf-tsp) to train & test our model withour installing all the dependencies by running: ```docker run -it --gpus all --rm fmsjung/lit-cnn-trnsf-tsp```

To train the proposed model on TSP20, run the following:
```console
$ python train.py --exp_name Experiment-Name -n 20 --bsz 512 --gpu_id 0 --embedding conv --nb_neighbors 10 --kernel_size 11 --segm_len 5 --batchnorm
```
To test the trained model on TSP20 random dataset, run the following:
```console
$ python test.py -n 20 --embedding conv --nb_neighbors 10 --kernel_size 11 --gpu_id 0 --ckpt_file Experiment-Name-on-Train.pkl --exp_tag Result-File-Name --segm_len 5 --greedy --beamsearch --batchnorm
```
OR you can instead run shell scripts for train and test respectively. You may modify the .sh files uploaded on this repository if necessary.
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

# Cite

```

@article{cite-key,
	author = {Jung, Minseop and Lee, Jaeseung and Kim, Jibum},
	date = {2024/06/19}
	doi = {10.1007/s10489-024-05603-x},
	id = {Jung2024},
	isbn = {1573-7497},
	journal = {Applied Intelligence},
	title = {A lightweight CNN-transformer model for learning traveling salesman problems},
	year = {2024}

```
