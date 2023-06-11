#!/bin/bash


class_name=${1}
num_iter=${2}
goal_time=${3}
dir_path=../formulas/${class_name}
core_dir_path=../dataset/${class_name}_core

for input in $dir_path/*.cnf; do
  cnf_name=${input##*/}
  cnf_name=${cnf_name%.cnf}
  cnf_name=${cnf_name%_repeat*}
  bash remove_core.sh $input ${core_dir_path}/${cnf_name}_core $num_iter $goal_time
done