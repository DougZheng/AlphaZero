#pragma once
#include "board.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>
#include <iomanip>
#include <random>
#include <tuple>

class Node {
public:
    friend class MCTS;
    Node() = default;
    Node(Node *parent, double p_sa);
    std::pair<Node*, int> select(double c_puct) const;
    void expand(const std::vector<double> &action_priors, const std::vector<int> &actions);
    void backup(double value);
    double get_value(double c_puct) const;
    bool is_leaf() const;
private:
    Node *parent = nullptr;
    std::vector<std::pair<Node*, int>> children;
    int n_visit = 0;
    double q_sa = 0;
    double p_sa = 1;
};

class MCTS {
public:
    MCTS(int n_playout, double c_puct = 5);
    static void destroy(Node *root);
    void playout(Board board);
    int get_move(const Board &board);
    std::pair<std::vector<double>, double> policy(Board &board);
    void display(Node *root, const Board &board) const;
private:
    std::unique_ptr<Node, decltype(MCTS::destroy)*> root;
    double c_puct;
    int n_playout;
};