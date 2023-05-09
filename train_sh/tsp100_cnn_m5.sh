python train.py \
--embedding conv \
--exp_name tsp100_cnn_m5 \
-n 100 \
--bsz 512 \
--gpu_id 0 \
--nb_neighbors 10 \
--kernel_size 11 \
--segm_len 5 \
--batchnorm