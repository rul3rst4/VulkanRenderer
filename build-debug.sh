#!/bin/bash

# Conan + CMake build script for Vulkan project

echo "Installing Conan dependencies..."
conan install . --output-folder=. --build=missing --settings=build_type=Debug --profile=clang20-cpp23-debug

echo "Configuring CMake..."
cmake --preset conan-debug

echo "Building project..."
cmake --build build/Debug

echo "Build complete! Executable is at: build/Debug/VulkanTutorial"
