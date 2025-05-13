# create a CMake project
cmake_minimum_required(VERSION 3.30.5)
project(VulkanTutorial)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HOMEBREW_PREFIX "/opt/homebrew/opt"
    CACHE PATH "Path to Homebrew installation")

set(CMAKE_PREFIX_PATH
    "${HOMEBREW_PREFIX}"
    "${HOMEBREW_PREFIX}/glm"
    "${HOMEBREW_PREFIX}/glfw"
    "/Users/andersonkulitch/VulkanSDK/1.4.309.0/macOS"
)

find_package(Vulkan REQUIRED)
include_directories(BEFORE ${Vulkan_INCLUDE_DIRS})
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(fmt REQUIRED)

list(TRANSFORM CMAKE_PREFIX_PATH APPEND "/include"
     OUTPUT_VARIABLE CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)

# set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# add the executable
add_executable(VulkanTutorial main.cpp)
target_compile_options(VulkanTutorial PUBLIC -Wall -Wextra -Werror -Wunused -Wpedantic)

target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES} Vulkan::Vulkan glm::glm fmt::fmt)
