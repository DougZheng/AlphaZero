#include "board.h"

std::vector<std::vector<int>> Board::dir = {
    {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

Board::Board(int start_player) : cur_player(start_player) {
    states.resize(n);
    len.resize(n);
    for (int i = 0; i < n; ++i) {
        states[i].resize(n);
        len[i].resize(n, std::vector<int>(8));
    }
}

void Board::exec_move(int pos) {
    return exec_move(pos / n, pos % n);
}

void Board::exec_move(int x, int y) {
    states[x][y] = cur_player;
    cur_player = -cur_player;
    last_move = x * n + y;
    ++cnt_move;
    auto is_same = [&](int x, int y, int aim) {
        return x >= 0 && x < n && y >= 0 && y < n && states[x][y] == aim;
    };
    for (int i = 0; i < 8; ++i) {
        int nx = x + dir[i][0], ny = y + dir[i][1];
        len[x][y][i] = 1;
        if (!is_same(nx, ny, states[x][y])) continue;
        int px = x + dir[i ^ 1][0], py = y + dir[i ^ 1][1];
        int plen = 1, nlen = len[nx][ny][i];
        if (is_same(px, py, states[x][y])) {
            plen += len[px][py][i ^ 1];
        } 
        len[x][y][i] += nlen;
        len[x + dir[i][0] * nlen][y + dir[i][1] * nlen][i ^ 1] += plen;
    }
    for (int i = 0; i < 8; i += 2) {
        if (len[x][y][i] + len[x][y][i ^ 1] - 1 >= n_in_row) {
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
    return cnt_move == get_board_size();
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