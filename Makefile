# MLX-LLM Makefile for M4 MacBook Pro
# Optimized for Apple Silicon with Metal GPU acceleration

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=armv8.5-a+fp16+bf16
LDFLAGS = -framework Metal -framework MetalPerformanceShaders -framework Accelerate

# Paths
MLX_PATH = /opt/homebrew/opt/mlx
HTTPLIB_PATH = /opt/homebrew/opt/cpp-httplib
JSON_PATH = /opt/homebrew/opt/nlohmann-json

# Include directories
INCLUDES = -I$(MLX_PATH)/include \
           -I$(HTTPLIB_PATH)/include \
           -I$(JSON_PATH)/include \
           -I.

# Library directories and libraries
LIBS = -L$(MLX_PATH)/lib -lmlx -lmlxnn

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
	@echo "Installing dependencies..."
	@command -v brew >/dev/null 2>&1 || { echo "Homebrew not found. Please install it first."; exit 1; }
	brew install llvm
	brew install nlohmann-json
	brew install cpp-httplib
	@echo "Installing MLX from source..."
	@if [ ! -d "mlx" ]; then \
		git clone https://github.com/ml-explore/mlx.git; \
		cd mlx && mkdir -p build && cd build; \
		cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_METAL=ON; \
		make -j8 && sudo make install; \
	fi
	@echo "Dependencies installed!"

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
	@echo "Usage examples:"
	@echo "  make              # Build the project"
	@echo "  make deps         # Install dependencies"
	@echo "  make install      # Install the binary"
	@echo "  make serve        # Run the server"
	@echo "  make m4-optimized # Build with M4 optimizations"

.PHONY: all clean deps install uninstall serve test debug m4-optimized help