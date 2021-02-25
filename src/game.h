#pragma once
#include "board.h"
#include <iostream>
#include <algorithm>
#include <vector>

class Game {
public:
    Game(int n, int n_in_row);
    void run(); 
private:
    Board board;
};