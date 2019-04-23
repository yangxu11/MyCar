#!/bin/bash
# get all filename in specified path

path=$1
files=$(ls $path)

mkdir 0
mkdir 1
mkdir 2
mkdir 3

for filename in $files
do
    if [ [ ${filename:0:1} -eq "0" ] ]; then
        mv $filename 0
    elif [ [ ${filename:0:1} -eq "1" ] ]; then
        mv $filename 1
    elif [ [ ${filename:0:1} -eq "2" ] ]; then
        mv $filename 2
    else
        mv $filename 3
    fi
done