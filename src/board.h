#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>

class Board {
public:
    Board(int n, int n_in_row, int cur_player = 1);
    void exec_move(int pos);
    void exec_move(int x, int y);
    std::vector<int> get_moves() const;
    std::pair<bool, int> get_result() const { return result; }
    bool is_legal(int pos) const;
    bool is_legal(int x, int y) const;
    void display() const;
    int get_cur_player() const { return cur_player; }
    int get_n() const { return n; }
    int get_board_size() const;
    bool get_is_tie() const;
    std::vector<std::vector<int>> get_states() const { return states; }
    int get_last_move() const { return last_move; }
    int get_move_cnt() const { return move_cnt; }
    std::vector<std::vector<std::vector<int>>> get_encode_states() const;
private:
    std::vector<std::vector<int>> states;
    int n = 15;
    int n_in_row = 5;
    int cur_player = 1;
    int last_move = -1;
    int move_cnt = 0;
    std::pair<bool, int> result{false, 0};
};