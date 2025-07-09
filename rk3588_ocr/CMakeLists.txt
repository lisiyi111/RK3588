cmake_minimum_required(VERSION 3.10)

project(stdeploy C CXX)

set(CMAKE_CXX_STANDARD 11)
set(STDEPLOY_VERSION_MAJOR "0")
set(STDEPLOY_VERSION_MINOR "0")
set(STDEPLOY_VERSION_PATCH "0")
set(LIBRARY_NAME "st")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## lib attribute
option(ENABLE_STDEPLOY_BUILD_SHARED "ENABLE_STDEPLOY_BUILD_SHARED" OFF)
## backend
option(ENABLE_RKNN_BACKEND "Whether to enable rknpu/rknpu2 backend" ON)
option(ENABLE_RKNPU2_BACKEND "Whether to enable rknpu2 backend" ON)
option(ENABLE_RKNPU2_RK3588 "Whether to enable RK3588 backend" ON)

## vison
option(ENABLE_VISION "Whether to enable vision model." ON)
option(ENABLE_VISION_CLASSIFICATION "Whether to enable vision classification model." ON)
option(ENABLE_VISION_OCR "Whether to enable vision ocr model." ON)

## set
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${ROOT_PATH}/workspace)

## install path
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif ()
set(STDEPLOY_INSTALL_PATH ${ROOT_PATH}/build/install)
if (ENABLE_STDEPLOY_BUILD_SHARED)
    set(STDEPLOY_INSTALL_TYPE LIBRARY)
else ()
    set(STDEPLOY_INSTALL_TYPE ARCHIVE)
endif ()
set(CMAKE_INSTALL_PREFIX ${STDEPLOY_INSTALL_PATH})
set(STDEPLOY_INSTALL_LIB_PATH ${STDEPLOY_INSTALL_PATH}/lib)
set(STDEPLOY_INSTALL_BIN_PATH ${STDEPLOY_INSTALL_PATH}/bin)
## build type
if (ENABLE_STDEPLOY_BUILD_SHARED)
    set(STDEPLOY_LIB_TYPE SHARED)
else ()
    set(STDEPLOY_LIB_TYPE STATIC)
endif ()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif ()

##  general
set(STDEPLOY_LIB_PREFIX "lib")
set(STDEPLOY_LIB_SUFFIX ".so")
if (CMAKE_SYSTEM_NAME MATCHES "^Android")
    set(SYSTEM.Android 1)
elseif (CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set(SYSTEM.Linux 1)
elseif (CMAKE_SYSTEM_NAME MATCHES "^Darwin")
    set(SYSTEM.Darwin 1)
    set(STDEPLOY_LIB_SUFFIX ".dylib")
elseif (CMAKE_SYSTEM_NAME MATCHES "^iOS")
    set(SYSTEM.iOS 1)
elseif (CMAKE_SYSTEM_NAME MATCHES "^Windows")
    set(SYSTEM.Windows 1)
    set(STDEPLOY_LIB_PREFIX "")
    set(STDEPLOY_LIB_SUFFIX ".lib")
endif ()

if (SYSTEM.Windows AND ENABLE_STDEPLOY_BUILD_SHARED)
    add_definitions(-DENABLE_STDEPLOY_BUILDING_DLL)
endif ()

if (!SYSTEM.Windows)
    # you must have return sentence for un-void function
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif ()

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif ()

add_definitions(-DSTDEPLOY_LIB)

##################################### Building: StDeploy C++ SDK #######################
add_definitions(-DSTDEPLOY_LIB)
include_directories(${ROOT_PATH})

## srcs
set(ALL_DEPLOY_SRCS "")
set(DEPEND_LIBS "")
set(SPECIFIC_LIBS "")
## base srcs
file(GLOB_RECURSE ALL_DEPLOY_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/stdeploy_model.cpp)
file(GLOB_RECURSE DEPLOY_CORE_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/core/*.cpp)
file(GLOB_RECURSE DEPLOY_RUNTIME_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/runtime/runtime.cpp
        ${PROJECT_SOURCE_DIR}/stdeploy/runtime/runtime_option.cpp
        ${PROJECT_SOURCE_DIR}/stdeploy/runtime/enum_variables.cpp)
file(GLOB_RECURSE DEPLOY_UTILS_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/utils/*.cpp)
list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_UTILS_SRCS})
list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_CORE_SRCS})
list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RUNTIME_SRCS})
## add rknn libs
if (ENABLE_RKNN_BACKEND)
    add_definitions(-DENABLE_RKNN_BACKEND)
    if (ENABLE_RKNPU2_BACKEND)
        add_definitions(-DENABLE_RKNPU2_BACKEND)
        file(GLOB_RECURSE DEPLOY_RKNN_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/runtime/backends/rknpu2/*.cpp)
        list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RKNN_SRCS})
        include(${PROJECT_SOURCE_DIR}/cmake/rknpu2.cmake)
    endif ()
endif ()
## add vision
if (ENABLE_VISION)#    ## add json
    include_directories(${PROJECT_SOURCE_DIR}/3rdparty/json)
    include_directories(${PROJECT_SOURCE_DIR}/3rdparty/json/single_include)
    ## add vision base srcs
    add_definitions(-DENABLE_VISION)
    file(GLOB_RECURSE DEPLOY_COMMON_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/vision/common/*.cpp)
    file(GLOB_RECURSE DEPLOY_VISUALIZE_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/vision/visualize/*.cpp)
    file(GLOB_RECURSE DEPLOY_VISION_UTILS_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/vision/utils/*.cpp)
    list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_COMMON_SRCS})
    list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_VISUALIZE_SRCS})
    list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_VISION_UTILS_SRCS})
    ## add vision classification
    if (ENABLE_VISION_CLASSIFICATION)
        add_definitions(-DENABLE_VISION_CLASSIFICATION)
        file(GLOB_RECURSE DEPLOY_CLS_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/vision/classification/*.cpp)
        list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_CLS_SRCS})
    endif ()
    ## add vision ocr
    if (ENABLE_VISION_OCR)
        add_definitions(-DENABLE_VISION_OCR)
        file(GLOB_RECURSE DEPLOY_OCR_SRCS ${PROJECT_SOURCE_DIR}/stdeploy/vision/ocr/*.cpp)
        list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_OCR_SRCS})
    endif ()
endif ()

## lib info
set(LIBRARY_NAME ${LIBRARY_NAME}_ai)
string(TIMESTAMP STDEPLOY_BUILD_DATE "%Y%m%d")
set(STDEPLOY_VERSION "v${STDEPLOY_VERSION_MAJOR}.${STDEPLOY_VERSION_MINOR}.${STDEPLOY_VERSION_PATCH}_beta_${STDEPLOY_BUILD_DATE}") # version code
configure_file(${PROJECT_SOURCE_DIR}/stdeploy/version/version.h.in
        ${PROJECT_SOURCE_DIR}/stdeploy/version/version.h)
add_library(${LIBRARY_NAME} ${STDEPLOY_LIB_TYPE} ${ALL_DEPLOY_SRCS})

foreach (item ${DEPEND_LIBS})
    message(STATUS "lib include ${item}")
endforeach ()
foreach (item ${SPECIFIC_LIBS})
    message(STATUS "lib include ${item}")
endforeach ()

## install
install(TARGETS ${LIBRARY_NAME} ${STDEPLOY_INSTALL_TYPE} DESTINATION ${STDEPLOY_INSTALL_LIB_PATH})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/stdeploy DESTINATION include FILES_MATCHING PATTERN "*.h")

add_executable(demo examples/main.cpp)
target_link_libraries(demo ${LIBRARY_NAME} ${DEPEND_LIBS} ${SPECIFIC_LIBS})
