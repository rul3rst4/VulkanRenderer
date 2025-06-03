#!/bin/bash

# Conan + CMake build script for Vulkan project

echo "Installing Conan dependencies..."
conan install . --output-folder=. --build=missing --settings=build_type=Release --profile=clang20-cpp23

echo "Configuring CMake..."
cmake --preset conan-release

echo "Building project..."
cmake --build build/Release

echo "Build complete! Executable is at: build/Release/VulkanTutorial"
