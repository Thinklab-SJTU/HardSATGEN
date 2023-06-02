#!/bin/bash

no=0
no1=1

cnf_file=${1}
origin_core=${2}
num_iter=${3}
goal_time=${4}
goal_time=`echo $goal_time|awk '{print $1}'`

cnf_name=${cnf_file##*/}
cnf_name=${cnf_name%.cnf}
save_path=${cnf_file%/*}_post
mkdir $save_path

drat_file=${cnf_file%.cnf}.drat
core_file=${cnf_file%.cnf}_core
save_file=$save_path/${cnf_name}_r$no1.cnf

cd ../postprocess/cadical/build
./cadical ../../$cnf_file --no-binary ../../$drat_file
cd ../../drat-trim
./drat-trim ../$cnf_file ../$drat_file -c ../$core_file
cd ../..
python src/postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core --add_var
rm $drat_file
rm $core_file


while [ "$no" -lt $num_iter ]; do
  no=$((no + 1))
  no1=$((no1 + 1))
  
  cnf_file=$save_file
  drat_file=${cnf_file%.cnf}.drat
  core_file=${cnf_file%.cnf}_core
  save_file=$save_path/${cnf_name}_r$no1.cnf

  cd postprocess/cadical/build
  ./cadical ../../$cnf_file --no-binary ../../$drat_file &>> ../log/${cnf_name%.cnf}.log
  time=`tail -n 7 ../log/${cnf_name%.cnf}.log`
  time=${time#*:}
  time=${time%%seconds*}
  time=`eval echo $time|awk '{print $1}'`
  if [ `echo "$time > $goal_time" | bc` -eq 1 ]
  then 
    break 1
  fi

  cd ../../drat-trim
  ./drat-trim ../$cnf_file ../$drat_file -c ../$core_file
  cd ../..
  python src/postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core
  
  rm $cnf_file
  rm $core_file
  rm $drat_file

done


