#!/bin/bash
cd ..
cd glucose/build && cmake .. && make && cd ../..

for file in generated/*; do
  [[ -e "$file" ]] || break
  ./glucose/build/glucose-simp ./generated/$file | grep '^c CPU time' >>record.txt
done
