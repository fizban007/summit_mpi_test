cmake_minimum_required(VERSION 3.8)
project(Aperture LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(test "main.cu")
target_link_libraries(test ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
