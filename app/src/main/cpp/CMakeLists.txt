cmake_minimum_required(VERSION 3.22.1)
#project("ncnnupscaler")

# Punta alla cartella cmake dell'ABI correntemente in fase di compilazione
#set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
#find_package(ncnn REQUIRED)

#add_library(ncnnupscaler SHARED ncnn-lib.cpp)
#add_library(
#        aiupscaler
#        SHARED
#        native-lib.cpp
#)

#find_library(log-lib log)
#find_library(jnigraphics-lib jnigraphics)
#find_library(vulkan-lib vulkan)

#target_link_libraries(ncnnupscaler
#        ncnn
#        aiupscaler
#        ${log-lib}
#        ${jnigraphics-lib}
#        ${vulkan-lib}
#)

project("aiupscaler")

add_library(
        aiupscaler
        SHARED
        native-lib.cpp
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
        aiupscaler
        ${log-lib}
        ${jnigraphics-lib}
)
