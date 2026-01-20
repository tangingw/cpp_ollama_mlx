# MLX-LLM Makefile for M4 MacBook Pro
# Optimized for Apple Silicon with Metal GPU acceleration

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=armv8.5-a+fp16+bf16
LDFLAGS = -framework Metal -framework MetalPerformanceShaders -framework Accelerate

# Paths
MLX_PYTHON_PATH := $(shell python3 -c "import mlx; import os; print(os.path.dirname(mlx.__file__))" 2>/dev/null || echo "")
HTTPLIB_PATH = /opt/homebrew/opt/cpp-httplib
JSON_PATH = /opt/homebrew/opt/nlohmann-json

# Include directories
INCLUDES = -I$(HTTPLIB_PATH)/include \
           -I$(JSON_PATH)/include \
           -I.

# Check if MLX Python is installed and use it
ifneq ($(MLX_PYTHON_PATH),)
    INCLUDES += -I$(MLX_PYTHON_PATH)/include
    LIBS = -L$(MLX_PYTHON_PATH)/lib -lmlx -lmlxnn
else
    # Fallback to Homebrew installation
    MLX_PATH = /opt/homebrew/opt/mlx
    INCLUDES += -I$(MLX_PATH)/include
    LIBS = -L$(MLX_PATH)/lib -lmlx -lmlxnn
endif

# Source files
SOURCES = main.cpp mlx_llm.cpp server.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = mlx-llm

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)
	@echo "Build complete! Binary: $(TARGET)"

# Compile source files
%.o: %.cpp mlx_llm.h
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete!"

# Install dependencies via Homebrew
deps:
	@echo "Checking for Xcode Command Line Tools..."
	@if ! xcode-select -p &> /dev/null; then \
		echo "Installing Xcode Command Line Tools..."; \
		xcode-select --install; \
		echo ""; \
		echo "⚠️  IMPORTANT: Complete the Xcode installation popup, then run 'make deps' again"; \
		exit 1; \
	fi
	@echo "✓ Xcode Command Line Tools found"
	@echo ""
	@echo "Accepting Xcode license (may require password)..."
	@sudo xcodebuild -license accept 2>/dev/null || true
	@echo ""
	@echo "Installing Homebrew dependencies..."
	@command -v brew >/dev/null 2>&1 || { echo "Homebrew not found. Please install it first."; exit 1; }
	@echo "Installing build tools..."
	brew install cmake
	brew install llvm
	@echo "Installing libraries..."
	brew install nlohmann-json
	brew install cpp-httplib
	@echo ""
	@echo "Installing MLX..."
	@echo "Note: MLX is best installed via pip. Installing Python MLX package..."
	pip3 install mlx
	@echo ""
	@echo "Note: For C++ MLX development, you have two options:"
	@echo ""
	@echo "Option 1 (Recommended): Use MLX Python bindings and FFI"
	@echo "  - Already installed via pip"
	@echo "  - Headers available in Python site-packages"
	@echo ""
	@echo "Option 2: Build MLX C++ from source manually"
	@echo "  git clone https://github.com/ml-explore/mlx.git"
	@echo "  cd mlx && mkdir -p build && cd build"
	@echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/homebrew"
	@echo "  make -j8 && sudo make install"
	@echo ""
	@echo "✓ Dependencies installed!"
	@echo ""
	@echo "If you want to use MLX C++ (Option 2), run the commands above manually."
	@echo "Otherwise, you can proceed with building using MLX Python bindings."

# Install the binary
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin..."
	sudo cp $(TARGET) /usr/local/bin/
	@echo "Installation complete!"

# Uninstall the binary
uninstall:
	@echo "Removing $(TARGET) from /usr/local/bin..."
	sudo rm -f /usr/local/bin/$(TARGET)
	@echo "Uninstall complete!"

# Run the server
serve: $(TARGET)
	./$(TARGET) serve

# Run tests (placeholder)
test: $(TARGET)
	@echo "Running tests..."
	./$(TARGET) list
	@echo "Tests complete!"

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -Wall -Wextra
debug: clean $(TARGET)
	@echo "Debug build complete!"

# Optimized build for M4
m4-optimized: CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=armv9-a+fp16+bf16+sve -mtune=apple-m4 -flto -ffast-math
m4-optimized: clean $(TARGET)
	@echo "M4-optimized build complete!"

# Show help
help:
	@echo "MLX-LLM Makefile"
	@echo "================"
	@echo ""
	@echo "Available targets:"
	@echo "  all            - Build the project (default)"
	@echo "  clean          - Remove build artifacts"
	@echo "  deps           - Install dependencies via Homebrew"
	@echo "  install        - Install binary to /usr/local/bin"
	@echo "  uninstall      - Remove binary from /usr/local/bin"
	@echo "  serve          - Build and run the server"
	@echo "  test           - Run tests"
	@echo "  debug          - Build with debug symbols"
	@echo "  m4-optimized   - Build with M4-specific optimizations"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Troubleshooting:"
	@echo "  fix-xcode      - Fix Xcode/Metal compiler issues"
	@echo ""
	@echo "Usage examples:"
	@echo "  make              # Build the project"
	@echo "  make deps         # Install dependencies"
	@echo "  make install      # Install the binary"
	@echo "  make serve        # Run the server"
	@echo "  make m4-optimized # Build with M4 optimizations"

# Clean build artifacts and MLX source
clean-all: clean
	@echo "Cleaning MLX source directory..."
	rm -rf mlx
	@echo "All clean!"

.PHONY: all clean deps install uninstall serve test debug m4-optimized help clean-all rebuild-mlx fix-xcode