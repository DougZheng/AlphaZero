#include "mcts.h"

Node::Node(Node *parent, double p_sa) : 
    parent(parent), p_sa(p_sa) { }

void Node::expand(const std::vector<double> &action_priors, const std::vector<int> &actions) {
    {
        std::lock_guard<std::mutex> lock(this->lock);
        if (is_leaf) {
            children.reserve(actions.size());
            for (const auto &pos : actions) {
                children.emplace_back(std::make_pair(new Node(this, action_priors[pos]), pos));
            }
            is_leaf = false;
        }
    }
}

void Node::backup(double value) {
    if (parent != nullptr) {
        parent->backup(-value);
    }
    --virtual_loss;
    unsigned n_visit = ++this->n_visit;
    {
        std::lock_guard<std::mutex> lock(this->lock);
        q_sa = ((n_visit - 1) * q_sa + value) / n_visit;
    }
}

std::pair<Node*, int> Node::select(double c_puct, double c_virtual_loss) {
    auto res = *max_element(children.begin(), children.end(), 
        [&](const std::pair<Node*, int> &a, const std::pair<Node*, int> &b) {
            return a.first->get_value(c_puct, c_virtual_loss) < b.first->get_value(c_puct, c_virtual_loss);
        }); 
    ++res.first->virtual_loss;
    return res;
}

double Node::get_value(double c_puct, double c_virtual_loss) const {
    unsigned n_visit = this->n_visit.load();
    double u_sa = c_puct * p_sa * sqrt(parent->n_visit.load()) / (1 + n_visit);
    double virtual_loss = c_virtual_loss * this->virtual_loss.load();
    return n_visit == 0 ? u_sa : (q_sa * n_visit - virtual_loss) / n_visit + u_sa;
}

MCTS::MCTS(size_t thread_num, int n_playout, double c_puct, double c_virtual_loss) : 
    n_playout(n_playout), c_puct(c_puct), c_virtual_loss(c_virtual_loss),
    root(new Node(nullptr, 1.0), MCTS::tree_deleter), 
    thread_pool(new ThreadPool(thread_num)) { }

void MCTS::tree_deleter(Node *root) {
    for (auto &child : root->children) {
        tree_deleter(child.first);
    }
    delete root;
    root = nullptr;
}

void MCTS::playout(Board board) {
    Node *cur = root.get();
    while (!cur->get_is_leaf()) {
        auto nxt = cur->select(c_puct, c_virtual_loss);
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
    // std::cout << "get_move" << std::endl;
    std::vector<std::future<void>> futures;
    for (int i = 0; i < n_playout; ++i) {
        auto future = thread_pool->commit(std::bind(&MCTS::playout, this, board));
        futures.emplace_back(std::move(future));
        // std::cout << i << " ?? " << std::endl;
    }
    // std::cout << "finish" << std::endl;
    for (int i = 0; i < futures.size(); ++i) {
        futures[i].wait();
    }
    // display(root.get(), board);
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
        static std::mt19937 rnd(time(0));
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
            child.first->get_value(c_puct, c_virtual_loss));
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