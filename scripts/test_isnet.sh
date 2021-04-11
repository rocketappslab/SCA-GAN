python test.py --dataroot ./fashion_data --pairLst ./fashion_data/fashion-resize-pairs-test.csv \
--checkpoints_dir ./checkpoints --results_dir ./results \
--name csV4M0_SPADEonlyencoder_csloss_lr1 --model scagan_is \
--phase test --dataset_mode keypoint --norm instance --batchSize 1 \
--resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip \
--which_model_netG CSGen --which_epoch 400 --display_id 0