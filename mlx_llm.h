#ifndef MLX_LLM_H
#define MLX_LLM_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace mlx_llm {

struct ModelConfig {
    std::string model_path;
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int n_ctx = 2048;
    int n_batch = 512;
    bool use_metal = true;
};

struct GenerationConfig {
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
    std::vector<std::string> stop_sequences;
};

class MLXModel {
public:
    virtual ~MLXModel() = default;
    virtual bool load(const std::string& path) = 0;
    virtual std::string generate(const std::string& prompt, const GenerationConfig& config) = 0;
    virtual std::vector<float> embed(const std::string& text) = 0;
    virtual void unload() = 0;
};

class MLXModelImpl : public MLXModel {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
public:
    MLXModelImpl();
    ~MLXModelImpl() override;
    
    bool load(const std::string& path) override;
    std::string generate(const std::string& prompt, const GenerationConfig& config) override;
    std::vector<float> embed(const std::string& text) override;
    void unload() override;
};

class ModelManager {
private:
    std::unordered_map<std::string, std::shared_ptr<MLXModel>> models_;
    
public:
    bool loadModel(const std::string& name, const std::string& path);
    bool unloadModel(const std::string& name);
    std::shared_ptr<MLXModel> getModel(const std::string& name);
    std::vector<std::string> listModels() const;
};

class Server {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
public:
    Server();
    ~Server();
    
    bool start(int port, const std::string& host = "127.0.0.1");
    void stop();
    void setModelManager(std::shared_ptr<ModelManager> manager);
};

} // namespace mlx_llm

#endif // MLX_LLM_H