#include "mlx_llm.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <ctime>

using json = nlohmann::json;

namespace mlx_llm {

struct Server::Impl {
    std::unique_ptr<httplib::Server> server;
    std::shared_ptr<ModelManager> model_manager;
    bool running = false;
    
    void setupRoutes() {
        // Generate endpoint
        server->Post("/api/generate", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "default");
                std::string prompt = body.value("prompt", "");
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    res.set_content(json{{"error", "Model not found"}}.dump(), "application/json");
                    return;
                }
                
                GenerationConfig config;
                config.max_tokens = body.value("max_tokens", 256);
                config.temperature = body.value("temperature", 0.7f);
                config.top_p = body.value("top_p", 0.9f);
                
                if (body.contains("stop")) {
                    config.stop_sequences = body["stop"].get<std::vector<std::string>>();
                }
                
                std::string response = model->generate(prompt, config);
                
                json result = {
                    {"model", model_name},
                    {"response", response},
                    {"done", true}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        
        // Ollama-compatible chat endpoint
        server->Post("/api/chat", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "default");
                auto messages = body["messages"];
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    res.set_content(json{{"error", "Model not found"}}.dump(), "application/json");
                    return;
                }
                
                // Build prompt from messages
                std::string prompt;
                for (const auto& msg : messages) {
                    std::string role = msg["role"];
                    std::string content = msg["content"];
                    prompt += role + ": " + content + "\n";
                }
                
                GenerationConfig config;
                config.max_tokens = body.value("max_tokens", 256);
                
                std::string response = model->generate(prompt, config);
                
                json result = {
                    {"model", model_name},
                    {"message", {
                        {"role", "assistant"},
                        {"content", response}
                    }},
                    {"done", true}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        
        // OpenAI-compatible chat completions endpoint
        server->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "gpt-3.5-turbo");
                auto messages = body["messages"];
                bool stream = body.value("stream", false);
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    json error_response = {
                        {"error", {
                            {"message", "Model not found"},
                            {"type", "invalid_request_error"},
                            {"code", "model_not_found"}
                        }}
                    };
                    res.set_content(error_response.dump(), "application/json");
                    return;
                }
                
                // Build prompt from messages
                std::string prompt;
                for (const auto& msg : messages) {
                    std::string role = msg["role"];
                    std::string content = msg["content"];
                    prompt += role + ": " + content + "\n";
                }
                
                GenerationConfig config;
                config.max_tokens = body.value("max_tokens", 256);
                config.temperature = body.value("temperature", 0.7f);
                config.top_p = body.value("top_p", 1.0f);
                
                if (body.contains("stop")) {
                    auto stop = body["stop"];
                    if (stop.is_string()) {
                        config.stop_sequences.push_back(stop.get<std::string>());
                    } else if (stop.is_array()) {
                        config.stop_sequences = stop.get<std::vector<std::string>>();
                    }
                }
                
                std::string response_text = model->generate(prompt, config);
                
                // OpenAI-compatible response format
                json result = {
                    {"id", "chatcmpl-" + std::to_string(std::time(nullptr))},
                    {"object", "chat.completion"},
                    {"created", std::time(nullptr)},
                    {"model", model_name},
                    {"choices", json::array({
                        {
                            {"index", 0},
                            {"message", {
                                {"role", "assistant"},
                                {"content", response_text}
                            }},
                            {"finish_reason", "stop"}
                        }
                    })},
                    {"usage", {
                        {"prompt_tokens", static_cast<int>(prompt.length() / 4)},
                        {"completion_tokens", static_cast<int>(response_text.length() / 4)},
                        {"total_tokens", static_cast<int>((prompt.length() + response_text.length()) / 4)}
                    }}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error_response = {
                    {"error", {
                        {"message", e.what()},
                        {"type", "server_error"},
                        {"code", "internal_error"}
                    }}
                };
                res.set_content(error_response.dump(), "application/json");
            }
        });
        
        // OpenAI-compatible completions endpoint
        server->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "gpt-3.5-turbo");
                std::string prompt = body.value("prompt", "");
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    json error_response = {
                        {"error", {
                            {"message", "Model not found"},
                            {"type", "invalid_request_error"},
                            {"code", "model_not_found"}
                        }}
                    };
                    res.set_content(error_response.dump(), "application/json");
                    return;
                }
                
                GenerationConfig config;
                config.max_tokens = body.value("max_tokens", 256);
                config.temperature = body.value("temperature", 0.7f);
                config.top_p = body.value("top_p", 1.0f);
                
                if (body.contains("stop")) {
                    auto stop = body["stop"];
                    if (stop.is_string()) {
                        config.stop_sequences.push_back(stop.get<std::string>());
                    } else if (stop.is_array()) {
                        config.stop_sequences = stop.get<std::vector<std::string>>();
                    }
                }
                
                std::string response_text = model->generate(prompt, config);
                
                json result = {
                    {"id", "cmpl-" + std::to_string(std::time(nullptr))},
                    {"object", "text_completion"},
                    {"created", std::time(nullptr)},
                    {"model", model_name},
                    {"choices", json::array({
                        {
                            {"text", response_text},
                            {"index", 0},
                            {"finish_reason", "stop"}
                        }
                    })},
                    {"usage", {
                        {"prompt_tokens", static_cast<int>(prompt.length() / 4)},
                        {"completion_tokens", static_cast<int>(response_text.length() / 4)},
                        {"total_tokens", static_cast<int>((prompt.length() + response_text.length()) / 4)}
                    }}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error_response = {
                    {"error", {
                        {"message", e.what()},
                        {"type", "server_error"},
                        {"code", "internal_error"}
                    }}
                };
                res.set_content(error_response.dump(), "application/json");
            }
        });
        
        // Ollama list models endpoint
        server->Get("/api/tags", [this](const httplib::Request& req, httplib::Response& res) {
            auto model_names = model_manager->listModels();
            
            json models = json::array();
            for (const auto& name : model_names) {
                models.push_back({
                    {"name", name},
                    {"modified_at", "2024-01-01T00:00:00Z"}
                });
            }
            
            json result = {{"models", models}};
            res.set_content(result.dump(), "application/json");
        });
        
        // OpenAI-compatible list models endpoint
        server->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
            auto model_names = model_manager->listModels();
            
            json models = json::array();
            for (const auto& name : model_names) {
                models.push_back({
                    {"id", name},
                    {"object", "model"},
                    {"created", std::time(nullptr)},
                    {"owned_by", "mlx-llm"},
                    {"permission", json::array()},
                    {"root", name},
                    {"parent", nullptr}
                });
            }
            
            json result = {
                {"object", "list"},
                {"data", models}
            };
            res.set_content(result.dump(), "application/json");
        });
        
        // Pull/Load model endpoint
        server->Post("/api/pull", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                std::string model_name = body.value("name", "");
                std::string model_path = body.value("path", "");
                
                if (model_manager->loadModel(model_name, model_path)) {
                    res.set_content(json{{"status", "success"}}.dump(), "application/json");
                } else {
                    res.status = 500;
                    res.set_content(json{{"error", "Failed to load model"}}.dump(), "application/json");
                }
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        
        // Delete model endpoint
        server->Delete("/api/delete", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                std::string model_name = body.value("name", "");
                
                if (model_manager->unloadModel(model_name)) {
                    res.set_content(json{{"status", "success"}}.dump(), "application/json");
                } else {
                    res.status = 404;
                    res.set_content(json{{"error", "Model not found"}}.dump(), "application/json");
                }
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        
        // Ollama embeddings endpoint
        server->Post("/api/embeddings", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "default");
                std::string text = body.value("prompt", "");
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    res.set_content(json{{"error", "Model not found"}}.dump(), "application/json");
                    return;
                }
                
                auto embedding = model->embed(text);
                
                json result = {
                    {"embedding", embedding}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        
        // OpenAI-compatible embeddings endpoint
        server->Post("/v1/embeddings", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                std::string model_name = body.value("model", "text-embedding-ada-002");
                auto input = body["input"];
                
                auto model = model_manager->getModel(model_name);
                if (!model) {
                    res.status = 404;
                    json error_response = {
                        {"error", {
                            {"message", "Model not found"},
                            {"type", "invalid_request_error"},
                            {"code", "model_not_found"}
                        }}
                    };
                    res.set_content(error_response.dump(), "application/json");
                    return;
                }
                
                json data_array = json::array();
                
                // Handle both string and array inputs
                if (input.is_string()) {
                    std::string text = input.get<std::string>();
                    auto embedding = model->embed(text);
                    data_array.push_back({
                        {"object", "embedding"},
                        {"embedding", embedding},
                        {"index", 0}
                    });
                } else if (input.is_array()) {
                    int index = 0;
                    for (const auto& text_item : input) {
                        std::string text = text_item.get<std::string>();
                        auto embedding = model->embed(text);
                        data_array.push_back({
                            {"object", "embedding"},
                            {"embedding", embedding},
                            {"index", index++}
                        });
                    }
                }
                
                json result = {
                    {"object", "list"},
                    {"data", data_array},
                    {"model", model_name},
                    {"usage", {
                        {"prompt_tokens", 0},
                        {"total_tokens", 0}
                    }}
                };
                
                res.set_content(result.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error_response = {
                    {"error", {
                        {"message", e.what()},
                        {"type", "server_error"},
                        {"code", "internal_error"}
                    }}
                };
                res.set_content(error_response.dump(), "application/json");
            }
        });
    }
};

Server::Server() : impl_(std::make_unique<Impl>()) {
    impl_->server = std::make_unique<httplib::Server>();
}

Server::~Server() {
    stop();
}

bool Server::start(int port, const std::string& host) {
    impl_->setupRoutes();
    
    std::cout << "Starting server on " << host << ":" << port << std::endl;
    
    impl_->running = true;
    return impl_->server->listen(host.c_str(), port);
}

void Server::stop() {
    if (impl_->running) {
        impl_->server->stop();
        impl_->running = false;
        std::cout << "Server stopped" << std::endl;
    }
}

void Server::setModelManager(std::shared_ptr<ModelManager> manager) {
    impl_->model_manager = manager;
}

} // namespace mlx_llm