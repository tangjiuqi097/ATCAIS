#!/bin/bash

conda env create -n py37 -f py37.yaml
conda activate py37
pip install cython
pip install tqdm

cd pytracking/mmdetection
sh ./compile.sh
python setup.py develop
cd ../..




echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done



