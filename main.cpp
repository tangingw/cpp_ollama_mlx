#include "mlx_llm.h"
#include <iostream>
#include <string>
#include <csignal>
#include <memory>
#include <sstream>
#include <vector>

using namespace mlx_llm;

std::unique_ptr<Server> g_server;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    if (g_server) {
        g_server->stop();
    }
    exit(signum);
}

// Forward declaration
void interactivePrompt(std::shared_ptr<MLXModel> model, const std::string& model_name);

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  serve              Start the API server (default)\n"
              << "  run <model> <prompt>  Run a model with a prompt\n"
              << "  load <name> <path>    Load a model from path\n"
              << "  list                   List loaded models\n"
              << "  --port <port>          Server port (default: 11434)\n"
              << "  --host <host>          Server host (default: 127.0.0.1)\n"
              << "  --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    
    std::string command = "serve";
    std::string host = "127.0.0.1";
    int port = 11434;
    
    // Parse command line arguments
    if (argc > 1) {
        command = argv[1];
    }
    
    if (command == "--help" || command == "-h") {
        printUsage(argv[0]);
        return 0;
    }
    
    // Parse additional arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        }
    }
    
    // Initialize model manager
    auto model_manager = std::make_shared<ModelManager>();
    
    if (command == "serve") {
        std::cout << "MLX-LLM Server - Optimized for M4 MacBook Pro\n";
        std::cout << "============================================\n\n";
        
        // Create and start server
        g_server = std::make_unique<Server>();
        g_server->setModelManager(model_manager);
        
        std::cout << "OpenAI-compatible API endpoints:\n";
        std::cout << "  POST /v1/chat/completions    - Chat completions\n";
        std::cout << "  POST /v1/completions         - Text completions\n";
        std::cout << "  GET  /v1/models              - List models\n";
        std::cout << "  POST /v1/embeddings          - Generate embeddings\n\n";
        
        std::cout << "Ollama-compatible API endpoints:\n";
        std::cout << "  POST /api/generate           - Generate text\n";
        std::cout << "  POST /api/chat               - Chat completion\n";
        std::cout << "  GET  /api/tags               - List models\n";
        std::cout << "  POST /api/pull               - Load model\n";
        std::cout << "  DELETE /api/delete           - Unload model\n";
        std::cout << "  POST /api/embeddings         - Generate embeddings\n\n";
        
        if (!g_server->start(port, host)) {
            std::cerr << "Failed to start server\n";
            return 1;
        }
        
    } else if (command == "run" && argc >= 3) {
        std::string model_name = argv[2];
        
        std::cout << "Loading model: " << model_name << "\n";
        
        // Load model (assuming model path is same as name for demo)
        if (!model_manager->loadModel(model_name, model_name)) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        
        auto model = model_manager->getModel(model_name);
        
        // Check if prompt was provided
        if (argc >= 4) {
            // Single prompt mode
            std::string prompt;
            for (int i = 3; i < argc; i++) {
                prompt += argv[i];
                if (i < argc - 1) prompt += " ";
            }
            
            GenerationConfig config;
            config.max_tokens = 512;
            config.temperature = 0.7f;
            
            std::cout << "\nGenerating response...\n\n";
            std::string response = model->generate(prompt, config);
            std::cout << response << "\n";
        } else {
            // Interactive mode
            interactivePrompt(model, model_name);
        }
        
    } else if (command == "load" && argc >= 4) {
        std::string model_name = argv[2];
        std::string model_path = argv[3];
        
        std::cout << "Loading model '" << model_name << "' from: " << model_path << "\n";
        
        if (model_manager->loadModel(model_name, model_path)) {
            std::cout << "Model loaded successfully\n";
        } else {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        
    } else if (command == "list") {
        auto models = model_manager->listModels();
        
        if (models.empty()) {
            std::cout << "No models loaded\n";
        } else {
            std::cout << "Loaded models:\n";
            for (const auto& name : models) {
                std::cout << "  - " << name << "\n";
            }
        }
        
    } else {
        std::cerr << "Invalid command\n\n";
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}