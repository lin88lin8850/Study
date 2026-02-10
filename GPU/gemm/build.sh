#!/bin/bash

set -e  # 任何命令失败时退出

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

echo "Project: $PROJECT_DIR"
echo ""

# 检查 build 目录是否存在
if [ -d "$BUILD_DIR" ]; then
    echo "Found existing build directory at: $BUILD_DIR"
    echo "Removing old build directory..."
    rm -rf "$BUILD_DIR"
    echo "✓ Cleaned"
fi

# 创建新的 build 目录
echo "Creating new build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "✓ Entered $BUILD_DIR"
echo ""

# 运行 CMake 配置
echo "Running CMake configure..."
cmake ..
echo "✓ CMake configure complete"
echo ""

# 编译
echo "Building project..."
make -j$(nproc)
echo "✓ Build complete"
echo ""
