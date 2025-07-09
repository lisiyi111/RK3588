CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_CXX_FLAGS "-Wall -O3  -lm -ldl ${CMAKE_C_FLAGS_PUBLIC}")

if (ENABLE_RKNPU2_BACKEND)
    if (ENABLE_RKNPU2_RK3588)

        add_definitions(-DENABLE_RKNPU2_RK3588)
        set(LIBRARY_NAME "${LIBRARY_NAME}_rk3588")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")
        set(CROSS_COMPILER "aarch64-linux-gnu-")

        ## rknpu2
        set(RKNN_TOOLKIT_PATH ${ROOT_PATH}/3rdparty/rknpu2)
        set(RKNN_RT_LIB ${RKNN_TOOLKIT_PATH}/lib/librknnrt.so)
        list(APPEND DEPEND_LIBS ${RKNN_RT_LIB})
        include_directories(${RKNN_TOOLKIT_PATH}/include)

        ## opencv
        set(OPENCV_DIRECTORY  ${ROOT_PATH}/3rdparty/opencv)
        include_directories(${OPENCV_DIRECTORY}/include)
        include_directories(${OPENCV_DIRECTORY}/include/opencv4)
        include_directories(${OPENCV_DIRECTORY}/include/opencv4/opencv2)
        set(OPENCV_LIB ${OPENCV_DIRECTORY}/lib/libopencv_world.so)
        list(APPEND DEPEND_LIBS ${OPENCV_LIB})

        ## rga
        set(RGA_DIRECTORY ${ROOT_PATH}/3rdparty/rga)
        include_directories(${RGA_DIRECTORY}/include)
        set(RGA_LIB ${RGA_DIRECTORY}/lib/librga.a)
        list(APPEND DEPEND_LIBS ${RGA_LIB})

        ## allocator need
        include_directories(${RGA_DIRECTORY}/allocator/include)
        file(GLOB_RECURSE allocator_src ${RGA_DIRECTORY}/allocator/src/*.cpp)
        list(APPEND ALL_DEPLOY_SRCS ${allocator_src})
        include_directories(${RGA_DIRECTORY}/libdrm/include)
        include_directories(${RGA_DIRECTORY}/libdrm/include/libdrm)

        ## yaml-cpp
        set(YAML_DIRECTORY ${ROOT_PATH}/3rdparty/yaml-cpp)
        set(YAML_LIB ${YAML_DIRECTORY}/lib/libyaml-cpp.a)
        list(APPEND DEPEND_LIBS ${YAML_LIB})
        include_directories(${YAML_DIRECTORY}/include)
        include_directories(${YAML_DIRECTORY}/include/yaml-cpp)
        ##
        list(APPEND SPECIFIC_LIBS pthread)
    else ()
        message("While -DENABLE_RKNPU2_BACKEND=ON, must define -DENABLE_RKNPU2_RV1106 or -DENABLE_RKNPU2_RK3588")
    endif ()
    set(CMAKE_C_COMPILER ${CROSS_COMPILER}gcc)
    set(CMAKE_CXX_COMPILER ${CROSS_COMPILER}g++)
    set(CMAKE_AS ${CROSS_COMPILER}as)
    set(CMAKE_AR ${CROSS_COMPILER}ar)
    set(CMAKE_NM ${CROSS_COMPILER}nm)
    set(CMAKE_LD ${CROSS_COMPILER}ld)
    set(CMAKE_STRIP ${CROSS_COMPILER}srtip)
    set(CMAKE_RANLIB ${CROSS_COMPILER}ranlib)

endif ()
