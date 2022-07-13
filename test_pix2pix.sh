set -ex

CUDA_VISIBLE_DEVICES=1 python test.py --name test_mask_model_mskloss --model smilegan --num_frames 9 \
--netG unet_128 --direction AtoB --dataset_mode video --norm batch --load_size 128 --crop_size 128 \
--dataroot  ./datasets/UTDAL_all/ \
--eval --dataset UTDAL \
--output_nc 3
