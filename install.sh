#!/bin/bash

# What directory am I in?
submit_dir=$(pwd)

# Locate ASE
ase_path=$(which ase)
cd "$(dirname "$ase_path")"
cd ..
echo "ASE Path is $(pwd)"

# Copy MC library
echo "Copying MC library to $(pwd)/ase"
cp -r $submit_dir/mc $(pwd)/ase
echo "Successfully added Monte Carlo to ASE!"
u
# Done!
cd $submit_dir

