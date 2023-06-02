cd glucose/build && cmake .. && make && cd ../..

for file in $(ls "generated")
do
    ./glucose/build/glucose-simp ./generated/$file | grep '^c CPU time' >> record.txt
done