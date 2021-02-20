#pragma once
#include "board.h"
#include "thread_pool.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>
#include <iomanip>
#include <random>
#include <tuple>
#include <thread>
#include <atomic>

class Node {
public:
    friend class MCTS;
    Node() = default;
    Node(Node *parent, double p_sa);
    std::pair<Node*, int> select(double c_puct, double c_virtual_loss);
    void expand(const std::vector<double> &action_priors, const std::vector<int> &actions);
    void backup(double value);
    double get_value(double c_puct, double c_virtual_loss) const;
    bool is_leaf() const;
private:
    Node *parent = nullptr;
    std::vector<std::pair<Node*, int>> children;
    std::atomic<unsigned> n_visit{0};
    double q_sa = 0;
    double p_sa = 1;
    std::atomic<int> virtual_loss{0};
    std::mutex lock;
};

class MCTS {
public:
    MCTS(size_t thread_num, int n_playout, double c_puct, double c_virtual_loss);
    static void tree_deleter(Node *root);
    void playout(Board board);
    int get_move(const Board &board);
    std::pair<std::vector<double>, double> policy(Board &board);
    void display(Node *root, const Board &board) const;
private:
    std::unique_ptr<Node, decltype(MCTS::tree_deleter)*> root;
    std::unique_ptr<ThreadPool> thread_pool;
    int n_playout;
    double c_puct;
    double c_virtual_loss;
};