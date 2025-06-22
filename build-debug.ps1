#!/usr/bin/env pwsh

# Conan + CMake build script for Vulkan project

Write-Host "Installing Conan dependencies..." -ForegroundColor Green
conan install . --output-folder=. --build=missing --settings=build_type=Debug --profile=clang20-cpp23-debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "Conan install failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Configuring CMake..." -ForegroundColor Green
cmake --preset conan-debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Building project..." -ForegroundColor Green
cmake --build build/Debug --preset conan-debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build complete! Executable is at: build/Debug/VulkanTutorial" -ForegroundColor Green
