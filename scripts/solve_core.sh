#!/bin/bash

dir_path=${1}
cd ..

for cnf in $dir_path/*.cnf; do
  echo $cnf
  cd postprocess/cadical/build
#   ./cadical ../../../${cnf} --no-binary &>> ../../../$dir_path/solve.log
  ./cadical ../../../${cnf} --no-binary ../../../${cnf%.cnf}.drat &>> ../../../$dir_path/solve.log
  echo ../../../$dir_path.log
  cd ../../drat-trim
  ./drat-trim ../../${cnf} ../../${cnf%.cnf}.drat -c ../../${cnf%.cnf}_core
  rm -rf ../../${cnf%.cnf}.drat
  cd ../..
done
