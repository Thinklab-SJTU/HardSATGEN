#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0

python eval/conversion_lcg.py --src dataset/train_formulas/ -s dataset/train_set_vig/ -a sat2vig
python eval/conversion_fg.py --src dataset/train_formulas/ -s dataset/train_set_fg/
python main_train.py --epoch_num 201 --data_name train_set_fg --core_flag --model ECC

python main_test.py --epoch_num 200 --data_name train_set_fg --core_flag --model ECC --epoch_load 200
# python eval/conversion.py --src graphs/GCN_3_32_preTrue_dropFalse_yield1_019501.120000_0.dat --store-dir formulas --action=lcg2sat
# python eval/conversion_lcg.py --src ../graphs/train_set_lcg_GCN_3_32_preTrue_dropFalse_yield1_coreTrue_0_200_0.dat --store-dir ../formulas --action=lcg2sat