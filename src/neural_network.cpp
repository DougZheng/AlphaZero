#include "neural_network.h"
#include <utility>

using namespace std::chrono_literals;

/*
    in torch1.2.0, module is ref instead of ptr
*/
NeuralNetwork::NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size) :
    module(torch::jit::load(model_path.c_str())), use_gpu(use_gpu), batch_size(batch_size), 
    running(true), loop(nullptr) {
    if (use_gpu) {
        module.to(at::kCUDA);
    }
    loop = std::make_unique<std::thread>([this]() {
        while (running.load()) {
            infer();
        }
    });
}

NeuralNetwork::~NeuralNetwork() {
    running.store(false);
    loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(const Board &board) {
    int n = board.get_n();
    const auto raw_states = board.get_encode_states();
    std::vector<int> states1D;
    for (const auto &vc1 : raw_states) {
        for (const auto &vc2 : vc1) {
            states1D.insert(states1D.end(), vc2.cbegin(), vc2.cend());
        }
    }
    // get input states
    torch::Tensor states = torch::from_blob(&states1D[0], {1, 4, n, n}, 
        torch::dtype(torch::kInt32)).toType(torch::kFloat32);
    std::promise<return_type> promise;
    auto res = promise.get_future();
    {
        std::lock_guard<std::mutex> lock(this->lock);
        tasks.emplace(std::make_pair(states, std::move(promise)));
        cv.notify_all();
    }
    return res;
}

void NeuralNetwork::infer() {
    std::vector<torch::Tensor> states;
    std::vector<std::promise<return_type>> promises;
    bool timeout = false;
    while (states.size() < batch_size && !timeout) {
        {
            std::unique_lock<std::mutex> lock(this->lock);
            if (cv.wait_for(lock, 1ms, [this]() {
                return tasks.size() > 0;
            })) {
                auto task = std::move(tasks.front());
                states.emplace_back(std::move(task.first));
                promises.emplace_back(std::move(task.second));
                tasks.pop();
            }
            else {
                timeout = true;
            }
        }
    }
    if (states.empty()) {
        return;
    }
    // prepare input
    std::vector<torch::jit::IValue> inputs{
        use_gpu ? torch::cat(states, 0).to(at::kCUDA) : torch::cat(states, 0)
    };
    // get result from nn
    auto res = module.forward(inputs).toTuple();
    // log_softmax probability, so exp() is needed
    torch::Tensor p_batch = res->elements()[0].toTensor().exp().toType(torch::kFloat32).to(at::kCPU);
    torch::Tensor v_batch = res->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);
    for (unsigned i = 0; i < promises.size(); ++i) {
        torch::Tensor p = p_batch[i];
        torch::Tensor v = v_batch[i];
        std::vector<double> prob(static_cast<float*>(p.data_ptr()), 
            static_cast<float*>(p.data_ptr()) + p.size(0));
        std::vector<double> value{v.item<float>()};
        return_type temp{std::move(prob), std::move(value)};
        promises[i].set_value(std::move(temp));
    }
}