#include "mcts.h"

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
        [&](const std::pair<Node*, int> &a, const std::pair<Node*, int> &b) {
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
        double value = res.second == 0 ? 0 : res.second == board.get_cur_player() ? 1 : -1;
        cur->backup(-value);
    }
}

int MCTS::get_move(const Board &board) {
    for (int i = 0; i < n_playout; ++i) {
        playout(board);
    }
    // display(root, board);
    return max_element(root->children.cbegin(), root->children.cend(), 
        [](const std::pair<Node*, int> &a, const std::pair<Node*, int> &b) {
            return a.first->n_visit < b.first->n_visit;
        })->second;
}

std::pair<std::vector<double>, double> MCTS::policy(Board &board) {
    auto actions = board.get_moves();
    std::vector<double> action_probs(board.get_board_size(), 1.0 / actions.size());
    int player = board.get_cur_player();
    auto res = board.get_result();
    while (!res.first) {
        static std::mt19937 rnd(2333);
        board.exec_move(actions[rnd() % actions.size()]);
        actions = board.get_moves();
        res = board.get_result();
    }
    double value = res.second == 0 ? 0 : res.second == player ? 1 : -1;
    return std::make_pair(action_probs, value);
}

void MCTS::display(Node *root, const Board &board) const {
    int n = board.get_n();
    using tridouble = std::tuple<double, double, double>;
    std::vector<std::vector<tridouble>> priors(n, std::vector<tridouble>(n));
    for (const auto &child : root->children) {
        priors[child.second / n][child.second % n] = std::make_tuple(
            1.0 * child.first->n_visit / root->n_visit,
            child.first->q_sa,
            child.first->get_value(c_puct));
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::get<0>(priors[i][j]) 
            << "(" << std::showpos << std::get<1>(priors[i][j]) << ","
            << std::get<2>(priors[i][j]) << ")"
            << std::noshowpos << " \n"[j == n - 1];
        }
    }
    std::cout << std::endl;
}