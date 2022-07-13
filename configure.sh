#! /bin/sh

echo "-Building CMake..."
rm -rf ${PWD}/build
mkdir ${PWD}/build
cmake -DCMAKE_BUILD_TYPE=Release -S ${PWD}/ -B ${PWD}/build 
echo "Finished!"
echo "--------------------"

echo "-Building MakeFile..."
make -C ${PWD}/build
echo "Finished!"
echo "--------------------"