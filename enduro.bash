CONFIG_PATH=configs/enduro.config
 for i in `seq 1 8`;
 do
   echo worker $i
   CUDA_VISIBLE_DEVICES=-1 python extract.py -c $CONFIG_PATH &
   sleep 1.0
 done

CUDA_VISIBLE_DEVICES=0 python vae_train.py -c configs/enduro.config
CUDA_VISIBLE_DEVICES=0 python series.py -c configs/enduro.config
CUDA_VISIBLE_DEVICES=0 python rnn_train.py -c configs/enduro.config
CUDA_VISIBLE_DEVICES=-1 python train.py -c configs/enduro.config