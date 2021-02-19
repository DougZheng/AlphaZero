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
            if (i <= n - n_in_row) {
                int k;
                for (k = 1; k < n_in_row && val == states[i + k][j]; ++k);
                if (k == n_in_row) return {true, val};
            }
            if (j <= n - n_in_row) {
                int k;
                for (k = 1; k < n_in_row && val == states[i][j + k]; ++k);
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

int Board::get_board_size() const {
    return n * n;
}

void Board::display() const {
    int last_x = last_move != -1 ? last_move / n : n;
    int last_y = last_move != -1 ? last_move % n : n;
    std::cout << std::setfill('0') << std::endl;
    for (int i = -1; i < n; ++i) {
        for (int j = -1; j < n; ++j) {
            if (i == -1 && j == -1) {
                std::cout << "    ";
            }
            else if (i == -1) {
                std::cout << std::setw(2) << j << "  ";
            }
            else if (j == -1) {
                std::cout << std::setw(2) << i << "   "; 
            }
            else{
                char ch = states[i][j] == 0 ? '.' : states[i][j] == 1 ? 'x' : 'o';
                if (i == last_x && j == last_y) {
                    std::cout << "\033[31;1m" << ch << "\033[0m" << "   ";
                }
                else{
                    std::cout << ch << "   ";
                }
            }
        }
        std::cout << "\n\n";
    }
    std::cout << std::setfill(' ') << std::endl;
}