#!/bin/bash

class=${1}
gen_num=${2}

# train && test
python main_train.py --epoch_num 201 --data_name ${class} --core_flag --model GCN
python main_test.py --epoch_num 200 --data_name ${class} --core_flag --model GCN --repeat $gen_num


# generate sat.cnf form from dat
mkdir ./formulas/$class
python eval/conversion_lcg.py --src graphs/${class}_GCN_coreTrue_1_1_${gen_num}.dat --store-dir formulas/$class --action=lcg2sat

# post-process
cd postprocess && bash auto_rmcore_multi.sh $class 200 500
