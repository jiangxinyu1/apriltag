#!/bin/bash
###
 # @Author: your name
 # @Date: 2021-11-24 12:12:42
 # @LastEditTime: 2021-11-24 12:13:02
 # @LastEditors: your name
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /apriltag/buildpc.sh
### 

echo "Configuring and building  on PC..."
if [ ! -d "buildPc" ]; then
    mkdir buildPc
fi
cd buildPc
#cmake .. -DCMAKE_TOOLCHAIN_FILE=/home/xinyu/workspace/360/OFei/360-0366-demo/arm64_linux_anker.cmake
cmake .. 
make -j4

