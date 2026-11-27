# colmap-gpumatch: SIFT feature matching for COLMAP databases on an NVIDIA GPU (see main.cpp). Built by .github/workflows/siteref-windows.yml.
cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0091 NEW)  # CMAKE_MSVC_RUNTIME_LIBRARY
project(colmap_gpumatch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# Nothing to install beside it: the C runtime and the CUDA runtime are linked in, the driver comes with the card
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
# Every major architecture CUDA 13 builds for (Turing to Blackwell: RTX 20xx onwards), plus PTX for later cards
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

add_executable(colmap-gpumatch main.cpp kernel.cu)
target_compile_definitions(colmap-gpumatch PRIVATE UNICODE _UNICODE NOMINMAX)
target_compile_options(colmap-gpumatch PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W4 /utf-8 /O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=/utf-8>)
# Windows' own SQLite (winsqlite3.dll, in every Windows 10 and 11)
target_link_libraries(colmap-gpumatch PRIVATE winsqlite3)
install(TARGETS colmap-gpumatch RUNTIME DESTINATION bin)
