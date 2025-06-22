cmake_minimum_required(VERSION 3.30.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(VulkanTutorial)

if(DEFINED ENV{HOME})
  set(USER_HOME_DIRECTORY $ENV{HOME})
elseif(DEFINED ENV{USERPROFILE})
  set(USER_HOME_DIRECTORY $ENV{USERPROFILE})
else()
  message(WARNING "Could not determine user's home directory.")
endif()

if(WIN32)
    set(VULKAN_SDK "C:/VulkanSDK/1.4.313.2")
    set(ENV{VULKAN_SDK} "C:/VulkanSDK/1.4.313.2")
elseif(APPLE)
    set(CMAKE_PREFIX_PATH
        "${USER_HOME_DIRECTORY}/VulkanSDK/1.4.309.0/macOS"
    )
elseif(UNIX)
    # Linux
    set(CMAKE_PREFIX_PATH
        "${USER_HOME_DIRECTORY}/VulkanSDK/1.4.309.0/x86_64"
    )
endif()



find_package(Vulkan REQUIRED)
include_directories(BEFORE ${Vulkan_INCLUDE_DIRS})

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)


# set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# add the executable
add_executable(VulkanTutorial main.cpp ScopedOneTimeCommandBuffer.cpp WindowManager.cpp VulkanCore.cpp VulkanCore.hpp CubemapOffscreenRender.cpp)
if(MSVC)
    target_compile_options(VulkanTutorial PUBLIC
        /W4                    # Equivalent to -Wall -Wextra
        /WX                    # Equivalent to -Werror
        /w14254                # Equivalent to -Wunused (unused variables)
        /w14263                # Equivalent to -Wunused (unreferenced functions)
        /permissive-           # Equivalent to -Wpedantic (strict conformance)
        /wd4996                # Equivalent to -Wno-deprecated-declarations
        /wd4201                # Equivalent to -Wno-missing-field-initializers (anonymous unions/structs)
    )
else()
    target_compile_options(VulkanTutorial PUBLIC
        -Wall -Wextra -Werror -Wunused -Wpedantic
        -Wno-reorder-init-list -Wno-c99-designator
        -Wno-missing-field-initializers -Wno-deprecated-declarations
    )
endif()

target_link_libraries(${PROJECT_NAME} glfw Vulkan::Vulkan glm::glm fmt::fmt)
