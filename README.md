# yolov7-tensorrtapi-cpp
Pengembangan fitur light object detection menggunakan tensor rt api sebagai salah satu topik riset ICAR ITS

### Workflow
your own model -> onxx -> tensor rt model -> inference engine -> output

### Utilities
1. NVIDIA DIVER
https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#ubuntu-lts

2. CUDA 11.3
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

3. CUDNN 8.6.0.163
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

4. TensorRT 8.0
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

### Run
1. mkdir build && cd build
2. cmake ..
3. make
4. ./main

