#pragma once
#include "board.h"
#include "thread_pool.h"
#include "neural_network.h"
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
#include <future>

class Node {
public:
    friend class MCTS;
    friend class AlphaZero;
    Node(Node *parent, double p_sa, int action);
    Node* select(double c_puct, double c_virtual_loss);
    void expand(const std::vector<double> &action_priors, const std::vector<int> &actions);
    void backup(double value);
    double get_value(double c_puct, double c_virtual_loss) const;
    bool get_is_leaf() const { return is_leaf.load(); }
private:
    Node *parent;
    std::vector<Node*> children;
    int action;
    double p_sa = 1;
    double q_sa = 0;
    std::atomic<unsigned> n_visit{0};
    std::atomic<int> virtual_loss{0};
    std::atomic<bool> is_parent{false};
    std::atomic<bool> is_leaf{true};
    std::mutex mutex_val;
};

class MCTS {
public:
    MCTS(size_t thread_num, int n_playout, double c_puct, double c_virtual_loss);
    int get_action(const Board &board);
    std::pair<std::vector<double>, double> policy(Board &board);
    void update_with_move(int last_action);
    // void display(Node *root, const Board &board) const;
protected:
    void startup(const Board &board);
    static void tree_deleter(Node *root);
    void playout(Board board);
    std::unique_ptr<Node, decltype(MCTS::tree_deleter)*> root;
    std::unique_ptr<ThreadPool> thread_pool;
    int n_playout;
    double c_puct;
    double c_virtual_loss;
};

class AlphaZero : public MCTS {
public:
    AlphaZero(NeuralNetwork *neural_network, size_t thread_num, int n_playout, double c_puct, double c_virtual_loss);
    std::pair<std::vector<double>, double> policy(const Board &board);
    std::vector<double> get_action_probs(const Board &board, double temp = 0.0);
private:
    NeuralNetwork *neural_network;
};