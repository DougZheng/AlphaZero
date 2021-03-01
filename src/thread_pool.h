#pragma once
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

/*
    A simple thread pool with c++11
*/
class ThreadPool {
public:
    using task_type = std::function<void()>;

    ThreadPool(size_t thread_num) : stop(false) {
        for (int i = 0; i < thread_num; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    task_type task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        task_cv.wait(lock, [this]() {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto commit(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("commit on stopped ThreadPool");
            }
            tasks.emplace([task]() {
                (*task)();
            });
            task_cv.notify_one();
            return res;
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        task_cv.notify_all();
        // wait for all threads
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
private:
    std::vector<std::thread> workers;
    std::queue<task_type> tasks;
    std::mutex queue_mutex;
    std::condition_variable task_cv;
    bool stop;
};