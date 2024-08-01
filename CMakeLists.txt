# CMakeList.txt : CMake project for PathTracer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project(PathTracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES
    main.cpp
    VulkanDebug.cpp VulkanDebug.h
    VulkanDevice.cpp VulkanDevice.h
    VulkanRaytracing.cpp VulkanRaytracing.h
    VulkanSwapchain.cpp VulkanSwapchain.h
    VulkanUtils.cpp VulkanUtils.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Libraries
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE
    external/glm-1.0.1
    external/stb
    external/tinyobjloader
)
add_subdirectory(external/glfw-3.4)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw)

# List of directories to copy
set(COPY_DIRS textures
			  models
			  shaders
			  spv
)

# Copy directories to the build directory
foreach(dir ${COPY_DIRS})
	file(COPY ${dir} DESTINATION ${CMAKE_BINARY_DIR})
endforeach()