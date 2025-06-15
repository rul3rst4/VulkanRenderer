cmake_minimum_required(VERSION 3.30.5)

project(VulkanTutorial)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED ENV{HOME})
  set(USER_HOME_DIRECTORY $ENV{HOME})
elseif(DEFINED ENV{USERPROFILE})
  set(USER_HOME_DIRECTORY $ENV{USERPROFILE})
else()
  message(WARNING "Could not determine user's home directory.")
endif()

set(CMAKE_PREFIX_PATH
    "${USER_HOME_DIRECTORY}/VulkanSDK/1.4.309.0/macOS"
)

find_package(Vulkan REQUIRED)
include_directories(BEFORE ${Vulkan_INCLUDE_DIRS})

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(fmt REQUIRED)

# set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# add the executable
add_executable(VulkanTutorial main.cpp ScopedOneTimeCommandBuffer.cpp WindowManager.cpp VulkanCore.cpp)
target_compile_options(VulkanTutorial PUBLIC -Wall -Wextra -Werror -Wunused -Wpedantic -Wno-reorder-init-list -Wno-c99-designator -Wno-missing-field-initializers -Wno-deprecated-declarations)

target_link_libraries(${PROJECT_NAME} glfw Vulkan::Vulkan glm::glm fmt::fmt)
