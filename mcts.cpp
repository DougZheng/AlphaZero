#include "mcts.h"
#include "board.h"

class Node {
public:
    friend class MCTS;
    Node() = default;
    Node(Node *par, double p);
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

Node::Node(Node *par, double p) : parent(par), p_sa(p) {
    
}

void Node::expand(const std::vector<double> &action_priors, const std::vector<int> &actions) {
    for (const auto &pos : actions) {
        children.emplace_back(std::make_pair(new Node(this, action_priors[pos]), pos));
    }
}

void Node::backup(double value) {
    if (parent != nullptr) {
        parent->backup(-value);
    }
    ++n_visit;
    q_sa = ((n_visit - 1) * q_sa + value) / n_visit;
}

std::pair<Node*, int> Node::select(double c_puct) const {
    return *max_element(children.cbegin(), children.cend(), 
        [&](const auto &a, const auto &b) {
            return a.first->get_value(c_puct) < b.first->get_value(c_puct);
        }); 
}

double Node::get_value(double c_puct) const {
    double u_sa = c_puct * p_sa * sqrt(parent->n_visit) / (1 + n_visit);
    return q_sa + u_sa;
}

bool Node::is_leaf() const {
    return children.empty();
}

class MCTS {
public:
    MCTS(int n = 4000);
    ~MCTS();
    void destroy(Node *root);
    void playout(Board board);
    int get_move(const Board &board);
    std::pair<std::vector<double>, double> policy(Board &board);
private:
    Node *root = new Node(nullptr, 1.0);
    double c_puct = 5;
    int n_playout;
};

MCTS::MCTS(int n) : n_playout(n) {

}

MCTS::~MCTS() {
    destroy(root);
}

void MCTS::destroy(Node *root) {
    for (auto &child : root->children) {
        destroy(child.first);
    }
    delete root;
    root = nullptr;
}

void MCTS::playout(Board board) {
    Node *cur = root;
    while (!cur->is_leaf()) {
        auto nxt = cur->select(c_puct);
        board.exec_move(nxt.second);
        cur = nxt.first;
    }
    auto res = board.get_result();
    if (!res.first) {
        auto actions = board.get_moves();
        auto pi = policy(board);
        cur->expand(pi.first, actions);
        cur->backup(-pi.second);
    }
    else {
        double value = res.second == board.get_cur_player() ? 1 : -1;
        cur->backup(-value);
    }
}

int MCTS::get_move(const Board &board) {
    for (int i = 0; i < n_playout; ++i) {
        playout(board);
    }
    return max_element(root->children.cbegin(), root->children.cend(), 
        [](const auto &a, const auto &b) {
            return a.first->n_visit < b.first->n_visit;
        })->second;
}

std::pair<std::vector<double>, double> MCTS::policy(Board &board) {
    auto actions = board.get_moves();
    std::vector<double> action_probs(actions.size(), 1.0 / actions.size());
    int player = board.get_cur_player();
    auto res = board.get_result();
    while (!res.first) {
        std::random_shuffle(actions.begin(), actions.end());
        board.exec_move(actions.back());
        actions = board.get_moves();
        res = board.get_result();
    }
    double value = res.second == 0 ? 0 : res.second == player ? 1 : -1;
    return std::make_pair(action_probs, value);
}