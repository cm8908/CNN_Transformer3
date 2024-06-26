python test.py \
-n 100 \
--embedding conv \
--nb_neighbors 10 \
--kernel_size 11 \
--gpu_id 2 \
--ckpt_file tsp100_cnn_m5.pkl \
--exp_tag tsp100_cnn_m5 \
--segm_len 5 \
--greedy \
--beamsearch \
--beam_width 2500 \
--batchnorm