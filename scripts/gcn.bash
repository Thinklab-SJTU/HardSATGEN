#!/bin/bash

cd ..

python eval/conversion_lcg.py --src dataset/train_formulas/ -s dataset/train_set_vig/ -a sat2vig
python eval/conversion_lcg.py --src dataset/train_formulas/ -s dataset/train_set_lcg/
python main_train.py --epoch_num 201 --data_name train_set_lcg --core_flag --model GCN

python main_test.py --epoch_num 200 --data_name train_set_lcg --core_flag --model GCN --epoch_load 200
# python eval/conversion.py --src graphs/GCN_3_32_preTrue_dropFalse_yield1_019501.120000_0.dat --store-dir formulas --action=lcg2sat
# python eval/conversion_lcg.py --src ../graphs/train_set_lcg_GCN_3_32_preTrue_dropFalse_yield1_coreTrue_0_200_0.dat --store-dir ../formulas --action=lcg2sat