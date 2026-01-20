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

// Interactive prompt implementation
void interactivePrompt(std::shared_ptr<MLXModel> model, const std::string& model_name) {
    std::cout << "\n╭─────────────────────────────────────────╮\n";
    std::cout << "│  Interactive Chat Mode                  │\n";
    std::cout << "│  Model: " << model_name << std::string(33 - model_name.length(), ' ') << "│\n";
    std::cout << "╰─────────────────────────────────────────╯\n\n";
    
    std::cout << "Commands:\n";
    std::cout << "  /bye or /exit  - Exit the chat\n";
    std::cout << "  /clear         - Clear conversation history\n";
    std::cout << "  /multiline     - Enter multiline mode (end with Ctrl+D)\n";
    std::cout << "  /help          - Show this help\n\n";
    
    std::vector<std::string> conversation_history;
    
    while (true) {
        std::cout << ">>> ";
        std::string input;
        std::getline(std::cin, input);
        
        // Check for EOF (Ctrl+D)
        if (std::cin.eof()) {
            std::cout << "\nGoodbye!\n";
            break;
        }
        
        // Trim whitespace
        input.erase(0, input.find_first_not_of(" \t\n\r"));
        input.erase(input.find_last_not_of(" \t\n\r") + 1);
        
        if (input.empty()) {
            continue;
        }
        
        // Handle commands
        if (input == "/bye" || input == "/exit") {
            std::cout << "\nGoodbye!\n";
            break;
        } else if (input == "/clear") {
            conversation_history.clear();
            std::cout << "Conversation history cleared.\n\n";
            continue;
        } else if (input == "/help") {
            std::cout << "\nCommands:\n";
            std::cout << "  /bye or /exit  - Exit the chat\n";
            std::cout << "  /clear         - Clear conversation history\n";
            std::cout << "  /multiline     - Enter multiline mode (end with Ctrl+D)\n";
            std::cout << "  /help          - Show this help\n\n";
            continue;
        } else if (input == "/multiline") {
            std::cout << "Enter multiline input (press Ctrl+D on a new line to finish):\n";
            std::stringstream multiline;
            std::string line;
            while (std::getline(std::cin, line)) {
                multiline << line << "\n";
            }
            std::cin.clear(); // Clear EOF state
            input = multiline.str();
            
            if (input.empty()) {
                continue;
            }
        }
        
        // Add user input to history
        conversation_history.push_back("User: " + input);
        
        // Build context from conversation history
        std::string prompt;
        for (const auto& msg : conversation_history) {
            prompt += msg + "\n";
        }
        prompt += "Assistant: ";
        
        // Generate response
        GenerationConfig config;
        config.max_tokens = 256;
        config.temperature = 0.7f;
        config.stop_sequences = {"\nUser:", "\n>>>"};
        
        std::cout << "\nA: ";
        std::string response = model->generate(prompt, config);
        
        // Extract just the assistant's response (remove the prompt)
        size_t assistant_start = response.find("Assistant: ");
        if (assistant_start != std::string::npos) {
            response = response.substr(assistant_start + 11); // Length of "Assistant: "
        }
        
        // Clean up response
        for (const auto& stop : config.stop_sequences) {
            size_t pos = response.find(stop);
            if (pos != std::string::npos) {
                response = response.substr(0, pos);
            }
        }
        
        std::cout << response << "\n\n";
        
        // Add assistant response to history
        conversation_history.push_back("Assistant: " + response);
        
        // Limit conversation history to last 10 exchanges
        if (conversation_history.size() > 20) {
            conversation_history.erase(conversation_history.begin(), 
                                     conversation_history.begin() + 2);
        }
    }
}

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