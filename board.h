#pragma once
#include <iostream>
#include <algorithm>
#include <vector>

class Board {
public:
    Board(int start_player = 1);
    void exec_move(int pos);
    void exec_move(int x, int y);
    std::vector<int> get_moves() const;
    std::pair<bool, int> get_result() const;
    bool is_legal(int pos) const;
    bool is_legal(int x, int y) const;
    void display() const;
    int get_cur_player() const { return cur_player; }
private:
    std::vector<std::vector<int>> states;
    int n = 8;
    int n_in_row = 5;
    int cur_player;
    int last_move = -1;
};