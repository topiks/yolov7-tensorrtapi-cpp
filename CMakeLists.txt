cmake_minimum_required(VERSION 3.0)

# create project
set(ProjectName "main")
project(${ProjectName})

# select build system and set compile options

# create executable file
add_executable(${ProjectName} 
                main.cpp
                
                include/image_processor/detection_engine.cpp
                include/image_processor/image_processor.cpp

                include/inference_helper/inference_helper_log.h
                include/inference_helper/inference_helper.cpp
                include/inference_helper/inference_helper_tensorrt.cpp

                include/inference_helper/tensorrt/common.h
                include/inference_helper/tensorrt/logger.cpp
                include/inference_helper/tensorrt/logging.h
                include/inference_helper/tensorrt/ErrorRecorder.h

                include/common_helper/common_helper.cpp
                include/common_helper/common_helper_cv.cpp
                include/common_helper/bounding_box.cpp
                include/common_helper/tracker.cpp
                include/common_helper/kalman_filter.h
                include/common_helper/simple_matrix.h
                include/common_helper/hungarian_algorithm.h

            )

# include opencv
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} PUBLIC ${OpenCV_LIBS})
message(STATUS "OPENCV VERSION: ${OpenCV_VERSION}")

# include cuda
find_package(CUDA)
target_include_directories(${ProjectName} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${ProjectName} PUBLIC ${CUDA_LIBRARIES} nvinfer
nvonnxparser
nvinfer_plugin
cudnn
)
target_link_directories(${ProjectName} PUBLIC /usr/local/cuda/lib64)
message(STATUS "CUDA VERSION: ${CUDA_VERSION}")

# find_package(CUDA)
# if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0") 
#     if(MSVC_VERSION)
#         target_link_directories(${ProjectName} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/bin)
#         target_link_directories(${ProjectName} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
#     else()
#         target_link_directories(${ProjectName} PUBLIC /usr/local/cuda/lib64)
#     endif()
# endif()
# if(CUDA_FOUND)
#     if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0") 
#         if(NOT $ENV{TensorRT_ROOT} STREQUAL "")
#             target_link_directories(${ProjectName} PUBLIC $ENV{TensorRT_ROOT}/lib)
#         endif()
#     endif()
#     target_link_libraries(${ProjectName} PRIVATE
#         ${CUDA_LIBRARIES}
#         nvinfer
#         nvonnxparser
#         nvinfer_plugin
#         cudnn
#     )
#     target_include_directories(${ProjectName} PUBLIC
#         ${CUDA_INCLUDE_DIRS}
#         tensorrt
#     )
#     if(NOT $ENV{TensorRT_ROOT} STREQUAL "")
#         target_include_directories(${ProjectName} PUBLIC $ENV{TensorRT_ROOT}/include)
#     endif()
#     add_definitions(-DINFERENCE_HELPER_ENABLE_TENSORRT)
#     message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# else()
#     message(WARNING, "Cannot find CUDA")
# endif()

