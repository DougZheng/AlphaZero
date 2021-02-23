#pragma once
#include "board.h"
#include <iostream>
#include <algorithm>
#include <vector>

class Game {
public:
    Game();
    void run(); 
private:
    Board board;
};