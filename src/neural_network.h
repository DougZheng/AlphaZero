#pragma once
#include "board.h"
#include <torch/script.h>
#include <vector>
#include <string>
#include <queue>
#include <future>
#include <atomic>
#include <memory>

class NeuralNetwork {
public:
    using return_type = std::vector<std::vector<double>>;

    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    ~NeuralNetwork();

    std::future<return_type> commit(const Board &board);

    void set_batch_size(unsigned batch_size) { this->batch_size = batch_size; }
private:
    using task_type = std::pair<torch::Tensor, std::promise<return_type>>;

    void infer();

    std::unique_ptr<std::thread> loop;
    std::atomic<bool> running;
    std::queue<task_type> tasks;
    std::mutex lock;
    std::condition_variable cv;
    torch::jit::script::Module module;
    unsigned batch_size;
    bool use_gpu;
};