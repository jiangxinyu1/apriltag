#!/bin/bash

echo "Configuring and building  ..."
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
#cmake .. -DCMAKE_TOOLCHAIN_FILE=/home/xinyu/workspace/360/OFei/360-0366-demo/arm64_linux_anker.cmake
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/.Toolchain/arm_toolchain.cmake
make -j4

