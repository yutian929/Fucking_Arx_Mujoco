#!/bin/bash
set -e

# Store root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing pybind11 ==="
cd "$ROOT_DIR/SDK/"

# Clone pybind11 if not exists
if [ -d "pybind11" ]; then
    echo "pybind11 already cloned, skipping clone..."
else
    echo "Cloning pybind11..."
    git clone https://github.com/pybind/pybind11.git
fi

# Build pybind11 (remove existing build if present)
cd pybind11
if [ -d "build" ]; then
    echo "Removing existing pybind11 build directory..."
    rm -rf build
fi
echo "Building pybind11..."
mkdir build && cd build && cmake .. && make && sudo make install

echo "=== Building bimanual python bindings ==="
cd "$ROOT_DIR/SDK/R5/py/ARX_R5_python/bimanual/"

# Remove existing build if present
if [ -d "build" ]; then
    echo "Removing existing bimanual build directory..."
    rm -rf build
fi
echo "Building bimanual..."
mkdir -p build && cd build && cmake .. && make && make install

echo "=== Setting PYTHONPATH ==="
cd "$ROOT_DIR/SDK/R5/py/ARX_R5_python/"
conda env config vars set PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== Installation complete ==="