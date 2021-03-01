#include "mcts.h"
#include <iostream>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <utility>
#include <random>
#include <future>

Node::Node(Node *parent = nullptr, double p_sa = 1.0, int action = -1) : 
    parent(parent), p_sa(p_sa), action(action) { }

/*
    expand a leaf node
    param action is all legal actions can be taken
*/
void Node::expand(const std::vector<double> &action_priors, const std::vector<int> &actions) {
    if (!is_parent.exchange(true)) { // if not been expanded
        children.reserve(actions.size()); 
        for (const auto &action : actions) {
            children.emplace_back(new Node(this, action_priors[action], action));
        }
        is_leaf.store(false); // expand finished
    }
}

/*
    propagate value from leaf to root
    update n_visit and q_sa 
*/
void Node::backup(double value) {
    if (parent != nullptr) {
        parent->backup(-value); // -value for the opposite player
    }
    --virtual_loss;
    unsigned n_visit = ++this->n_visit;

    // simple mutex, have not consider the consistency of q_sa and n_visit
    // the side effect can be ignored
    { 
        std::lock_guard<std::mutex> lock(mutex_val);
        q_sa = ((n_visit - 1) * q_sa + value) / n_visit;
    }
}

/*
    select the next action
*/
Node* Node::select(double c_puct, double c_virtual_loss) {
    auto res = *max_element(children.begin(), children.end(), 
        [&](const Node *a, const Node *b) {
            return a->get_value(c_puct, c_virtual_loss) < b->get_value(c_puct, c_virtual_loss);
        }); 
    ++res->virtual_loss;
    return res;
}

/*
    evaluate an action
*/
double Node::get_value(double c_puct, double c_virtual_loss) const {
    unsigned n_visit = this->n_visit.load();
    double u_sa = c_puct * p_sa * sqrt(parent->n_visit.load()) / (1 + n_visit);
    double virtual_loss = c_virtual_loss * this->virtual_loss.load();
    return n_visit == 0 ? u_sa : (q_sa * n_visit - virtual_loss) / n_visit + u_sa;
}

MCTS::MCTS(size_t thread_num, int n_playout, double c_puct, double c_virtual_loss) : 
    n_playout(n_playout), c_puct(c_puct), c_virtual_loss(c_virtual_loss),
    root(new Node(), MCTS::tree_deleter), 
    thread_pool(new ThreadPool(thread_num)) { }

void MCTS::tree_deleter(Node *root) {
    for (auto &child : root->children) {
        tree_deleter(child);
    }
    delete root;
    root = nullptr;
}

/*
    simulation from root to leaf
    copy of board is needed
*/
void MCTS::playout(Board board) {
    Node *cur = root.get();
    while (!cur->get_is_leaf()) {
        cur = cur->select(c_puct, c_virtual_loss);
        board.exec_move(cur->action);
    }
    auto res = board.get_result();
    if (!res.first) { // if not ended
        auto actions = board.get_moves();
        auto pi = policy(board);
        cur->expand(pi.first, actions);
        // you may feel confused about the negative sign
        // recall that from parent's perspective, this cur node is represent for the oppoent
        cur->backup(-pi.second);
    }
    else {
        double value = res.second == 0 ? 0 : res.second == board.get_cur_player() ? 1 : -1;
        cur->backup(-value);
    }
}

/*
    get action, take action greedily 
*/
int MCTS::get_action(const Board &board) {
    startup(board);
    return (*max_element(root->children.cbegin(), root->children.cend(), 
        [](const Node *a, const Node *b) {
            return a->n_visit < b->n_visit;
        }))->action;
}

/*
    helper function
    start up all simulations
*/
void MCTS::startup(const Board &board) {
    int n_need = n_playout - root->n_visit; // reuse the previous result
    std::vector<std::future<void>> futures;
    futures.reserve(n_need);
    // commit all simulation tasks
    for (int i = 0; i < n_need; ++i) {
        auto future = thread_pool->commit(std::bind(&MCTS::playout, this, board));
        futures.emplace_back(std::move(future));
    }
    // wait for all tasks finished
    for (int i = 0; i < futures.size(); ++i) {
        futures[i].wait();
    }
}

/*
    naive policy, using rollout
    it is O(board_size), efficient enough
    return (action_priors, value)
*/
std::pair<std::vector<double>, double> MCTS::policy(Board &board) {
    auto actions = board.get_moves();
    // each action is taken in equal probability
    std::vector<double> action_priors(board.get_board_size(), 1.0 / actions.size());
    int player = board.get_cur_player();
    // take action randomly
    std::random_shuffle(actions.begin(), actions.end());
    std::pair<bool, int> res;
    for (const auto &act : actions) {
        board.exec_move(act);
        res = board.get_result();
        if (res.first) break;
    }
    double value = res.second == 0 ? 0 : res.second == player ? 1 : -1;
    return std::make_pair(action_priors, value);
}

/*
    update with the oppoent's move 
    reuse the tree instead of destroying it directly 
*/
void MCTS::update_with_move(int last_action) {
    for (auto it = root->children.begin(); it != root->children.end(); ++it) {
        if ((*it)->action == last_action) {
            Node *new_root = *it;
            root->children.erase(it);
            new_root->parent = nullptr;
            root.reset(new_root);
            return;
        }
    }
    // if there is not such action, reset the tree
    root.reset(new Node());
}

AlphaZero::AlphaZero(NeuralNetwork *neural_network, size_t thread_num, int n_playout, double c_puct, double c_virtual_loss) : 
    MCTS(thread_num, n_playout, c_puct, c_virtual_loss), neural_network(neural_network) { }

/*
    AlphaZero's policy, using neural network
*/
std::pair<std::vector<double>, double> AlphaZero::policy(Board &board) {
    auto future = neural_network->commit(board);
    auto result = future.get();
    double value = result[1][0];
    auto action_priors = std::move(result[0]);
    double sum = std::accumulate(action_priors.cbegin(), action_priors.cend(), 0.0);
    if (sum < FLT_EPSILON) {
        std::cerr << "Warning: no valid move." << std::endl;
        std::for_each(action_priors.begin(), action_priors.end(), 
            [sum](double &x) { x /= sum; });
    }
    else {
        // normalization
        std::for_each(action_priors.begin(), action_priors.end(), 
            [sum](double &x) { x /= sum; });
    }
    return std::make_pair(action_priors, value);
}

/*
    get action probs
*/
std::vector<double> AlphaZero::get_action_probs(const Board &board, double temp) {
    startup(board);
    std::vector<double> action_probs(board.get_board_size(), 0.0);
    if (temp < FLT_EPSILON) { // greedy
        int action = (*max_element(root->children.cbegin(), root->children.cend(), 
            [](const Node *a, const Node *b) {
                return a->n_visit < b->n_visit;
            }))->action;
        action_probs[action] = 1.0;
    }
    else {
        double sum = 0;
        // p:=p^(1/temp)
        for (const auto &child : root->children) {
            action_probs[child->action] = std::pow(child->n_visit.load(), 1.0 / temp);
            sum += action_probs[child->action];
        }
        // normalization
        std::for_each(action_probs.begin(), action_probs.end(), 
            [sum](double &x) { x /= sum; });
    }
    return action_probs;
}