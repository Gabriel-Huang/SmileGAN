set -ex


CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/UTDAL_all/ --dataset UTDAL --name test_mask_model_mskloss --label 'Happiness' \
--model smilegan --direction AtoB --lambda_L1 100 --dataset_mode video \
--norm batch --pool_size 0 --num_frames 9 \
--load_size 128 --crop_size 128 --netG unet_128 \
--display_freq 320 --print_freq 320 \
--niter 25 \
--niter_decay 15 \
--output_nc 3 \
