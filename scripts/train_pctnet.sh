python train.py --dataroot ./fashion_data --pairLst ./fashion_data/fashion-resize-pairs-train.csv \
--name scagan_pctnet --model scagan_pct --which_model_netG EdgeGen \
--lambda_GAN 5 --lambda_A 1 --lambda_B 1 --display_id 0 \
--dataset_mode edge --n_layers 3 --norm instance  --pool_size 0 --resize_or_crop no \
--gpu_ids 0 --batchSize 5 --BP_input_nc 18 --no_flip  \
--niter 2 --niter_decay 2 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 \
--with_D_PP 1 --with_D_PB 1 --use_cxloss 0 --print_freq 10 --continue_train 0 \
--lr 0.0002 --display_freq 100 --crop_radius 20
