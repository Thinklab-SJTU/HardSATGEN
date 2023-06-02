#!/bin/bash
# param
maxNumJobs=20

mkdir -p log

class_name=${1}
num_iter=${2}
goal_time=${3}
dir_path=../formulas/${class_name}

for input in $dir_path/*.cnf; do
  echo $input
  cnf_name=${input##*/}
  cnf_name=${cnf_name%.cnf}
  bash loose_core.sh $input $num_iter $goal_time &

  runningJobs=`jobs -r | wc -l`
  echo "Running Jobs: ${runningJobs}"

  while [ `jobs -r | wc -l` -ge ${maxNumJobs} ]; do
    :
  done
done


while [ `jobs -r | wc -l` -gt 0 ]; do
  :
done

echo "Finish all jobs:"
jobs
echo "========================"
