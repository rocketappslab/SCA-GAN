python train.py --dataroot ./fashion_data --pairLst ./fashion_data/fashion-resize-pairs-train.csv \
--name scagan_isnet --model scagan_is --which_model_netG CSGen \
--lambda_GAN 5 --lambda_A 1 --lambda_B 1 --lambda_cx 0.1 --lambda_AttriLoss 0.1 --display_id 0 \
--dataset_mode keypoint --n_layers 3 --norm instance  --pool_size 0 --resize_or_crop no \
--gpu_ids 0 --batchSize 3 --BP_input_nc 18 --no_flip  \
--niter 200 --niter_decay 200 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 \
--with_D_PP 1 --with_D_PB 1 --use_cxloss 1 --print_freq 10 --continue_train 0 \
--lr 1e-4 --display_freq 10 --use_vae True
