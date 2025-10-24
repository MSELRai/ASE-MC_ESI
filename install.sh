#!/bin/bash

# What directory am I in?
submit_dir=$(pwd)

# Locate ASE
#ase_executable_path=$(which ase)
#cd "$(dirname "$ase_executable_path")"
#cd ..
#ase_path=$(pwd)
#echo "ASE Path is $(ase_path)"

# Copy MC library
#echo "Copying MC library to $(pwd)/ase"
#cp -r $submit_dir/mc $(pwd)/ase
#echo "Successfully added Monte Carlo to ASE!"

# Locate ASE
ase_path="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cd "$ase_path/ase"
echo "ASE Path is $(pwd)"

# Copy MC library
echo "Copying MC library to $(pwd)"
cp -r $submit_dir/mc $(pwd)

echo "Successfully added Monte Carlo to ASE!"

# Done!
cd $submit_dir

