#!/usr/bin/env bash

cd utils/pyvotkit
python setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../

cd utils/nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=$1
cd ../../
python build.py
cd ../../

cd utils/roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=$1
cd ../../
python build.py
cd ../../../
