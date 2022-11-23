#!/bin/bash

os=$(uname)
if [ ${os} = "Linux" ]; then
    export LD_LIBRARY_PATH=$PWD/../machinelearning/artifacts/bin/Native/x64.Debug:$LD_LIBRARY_PATH
elif [ ${os} = "Darwin" ]; then
    export LIBRARY_PATH=$PWD/../machinelearning/artifacts/bin/Native/x64.Debug:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/../machinelearning/artifacts/bin/Native/x64.Debug:$DYLD_LIBRARY_PATH
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
    # TODO: fix linking on MacOS
    ln -s libOneDalNative.dylib ../machinelearning/artifacts/bin/Native/x64.Debug/libOneDalNative
    ln -s libonedal_core.1.1.dylib ../machinelearning/artifacts/bin/Native/x64.Debug/libonedal_core.dylib
    ln -s libonedal_thread.1.1.dylib ../machinelearning/artifacts/bin/Native/x64.Debug/libonedal_thread.dylib
else
    echo "Wrong system: ${os}"
    exit 1
fi
export CPATH=$CONDA_PREFIX/include:$CPATH
export PATH=$PWD/../machinelearning/.dotnet:$PATH
