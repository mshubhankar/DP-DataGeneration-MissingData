cmake_minimum_required(VERSION 3.5)
project(privbayes_new)

set(CMAKE_CXX_STANDARD 14)

include_directories(code)
include_directories(data)

add_executable(privbayes_new
        code/main_marginal.cpp
        code/main_svm.cpp
        code/methods.cpp
        code/methods.h
        code/noise.cpp
        code/noise.h
        code/printer.cpp
        code/printer.h
        code/table.cpp
        code/table.h
        code/tools.h
        code/translator.cpp
        code/translator.h)
