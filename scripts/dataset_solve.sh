#!/bin/bash

dir_path=${1}
cd ..

for cnf in $dir_path/*.cnf; do
  echo $cnf
  cd postprocess/cadical/build
  ../../../../runlim-master/runlim ./cadical ../../../${cnf} --no-binary &>> ../../../$dir_path/solve.log
  cd ../../..
done
