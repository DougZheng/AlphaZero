#include "board.h"

Board::Board(int n, int n_in_row, int cur_player) :
    n(n), n_in_row(n_in_row), cur_player(cur_player) {
    states.resize(n, std::vector<int>(n));
}

void Board::exec_move(int pos) {
    return exec_move(pos / n, pos % n);
}

void Board::exec_move(int x, int y) {
    states[x][y] = cur_player;
    cur_player = -cur_player;
    last_move = x * n + y;
    ++move_cnt;
    const static int dir[4][2] = {
        {0, 1}, {1, 0}, {1, 1}, {1, -1}
    };
    auto is_same = [&](int x, int y, int aim) {
        return x >= 0 && x < n && y >= 0 && y < n && states[x][y] == aim;
    };
    for (int i = 0; i < 4; ++i) {
        int len = 1, step = 1;
        while (is_same(x + step * dir[i][0], y + step * dir[i][1], states[x][y])) {
            ++step, ++len;
        }
        step = 1;
        while (is_same(x - step * dir[i][0], y - step * dir[i][1], states[x][y])) {
            ++step, ++len;
        }
        if (len >= n_in_row) {
            result = std::make_pair(true, states[x][y]);
            return;
        }
    }
    result = {get_is_tie(), 0};
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

bool Board::is_legal(int pos) const {
    return is_legal(pos / n, pos % n);
}

bool Board::is_legal(int x, int y) const {
    return states[x][y] == 0;
}

int Board::get_board_size() const {
    return n * n;
}

bool Board::get_is_tie() const {
    return move_cnt == get_board_size();
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

std::vector<std::vector<std::vector<int>>> Board::get_encode_states() const {
    std::vector<std::vector<std::vector<int>>> res(
        4, std::vector<std::vector<int>>(n, std::vector<int>(n)));
    int is_first = move_cnt % 2 == 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (states[i][j] == 1) res[0][i][j] = 1;
            else if(states[i][j] == -1) res[1][i][j] = 1;
            res[3][i][j] = is_first;
        }
    }
    if (last_move != -1) {
        res[2][last_move / n][last_move % n] = 1;
    }
    return res;
}