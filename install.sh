#!/bin/bash
# ROOT DIR
# Install pybind11
cd SDK/
git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make && sudo make install
cd ../../../

# Build bimanual python bindings
cd SDK/R5/py/ARX_R5_python/bimanual/
mkdir -p build && cd build && cmake .. && make && make install
cd ../../../../../../
cd SDK/R5/py/ARX_R5_python/
conda env config vars set PYTHONPATH="$PWD:$PYTHONPATH"
cd ../../../