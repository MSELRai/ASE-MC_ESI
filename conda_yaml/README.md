All of the simulations in the ASE-MC manuscript can be done with the above conda environments.
Some editing with these files is required on the "prefix" line on each of the yaml files.
You should change this line to the directory where you want the conda environment installed.

Any of the environments can be created with the above files with the following command at the terminal:
conda env create -f environment.yaml -n my_environment_name
