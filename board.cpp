#include "board.h"

Board::Board(int start_player) : cur_player(start_player) {
    states.resize(n);
    for (auto &vc : states) vc.resize(n);
}

void Board::exec_move(int pos) {
    exec_move(pos / n, pos % n);
}

void Board::exec_move(int x, int y) {
    states[x][y] = cur_player;
    cur_player = -cur_player;
    last_move = x * n + y;
}

std::vector<int> Board::get_moves() const {
    std::vector<int> moves;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (states[i][j] == 0) {
                moves.emplace_back(i * n + j);
            }
        }
    }
    return moves;
}

std::pair<bool, int> Board::get_result() const {
    bool can_move = false;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int val = states[i][j];
            if (val == 0) {
                can_move = true;
                continue;
            } 
            if (j <= n - n_in_row) {
                int k;
                for (k = 1; k < n_in_row && val == states[i][j + k]; ++k);
                if (k == n_in_row) return {true, val};
            }
            if (i <= n - n_in_row) {
                int k;
                for (k = 1; k < n_in_row && val == states[i + k][j]; ++k);
                if (k == n_in_row) return {true, val};
            }
            if (i <= n - n_in_row && j <= n - n_in_row) {
                int k;
                for (k = 1; k < n_in_row && val == states[i + k][j + k]; ++k);
                if (k == n_in_row) return {true, val};
            }
            if (i <= n - n_in_row && j >= n_in_row - 1) {
                int k;
                for (k = 1; k < n_in_row && val == states[i + k][j - k]; ++k);
                if (k == n_in_row) return {true, val};
            }
        }
    }
    return {!can_move, 0};
}

bool Board::is_legal(int pos) const {
    return is_legal(pos / n, pos % n);
}

bool Board::is_legal(int x, int y) const {
    return states[x][y] == 0;
}

void Board::display() const {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            char ch = states[i][j] == 0 ? '.' : states[i][j] == 1 ? 'x' : 'o';
            std::cout << ch << " \n"[j == n - 1];
        }
    }
    std::cout << std::endl;
}