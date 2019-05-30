#!/bin/sh

input_folder=$1
output_folder=$2

#compressions=( 10, 20, 30, 50 )

for file in $input_folder/*.txt; do
    for compression in 10 20 30 50; do
        python3 main.py -i "$file" -o "$output_folder/WOWO_$(basename -- $file)_$compression.txt" -c $compression --ranking="wowo" --case_insensitive;
        python3 main.py -i "$file" -o "$output_folder/WOKW_$(basename -- $file)_$compression.txt" -c $compression --ranking="wokw" --case_insensitive;
    done
done