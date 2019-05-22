#!/bin/bash

if [ -z "$1" ]; then
    echo "team name is required.
Usage: ./CollectSubmission team_*"
    exit 0
fi

files="./model_checkpoints/*.meta
./utils/data_utils.py
Assignment3.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done

rm -f $1.tar.gz
mkdir $1
cp -r ./model_checkpoints ./Assignment3.ipynb $1/
mkdir $1/utils
cp -r ./utils/data_utils.py $1/utils/
tar cvzf $1.tar.gz $1
rm -rf ./$1
