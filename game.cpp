#include "game.h"

Game::Game() {

}

void Game::run() {
    while (true) {
        board.display();
        std::cout << "now for " << (board.get_cur_player() == 1 ? 'x' : 'o') << ": ";
        int x, y;
        std::cin >> x >> y;
        while (!board.is_legal(x, y)) {
            std::cin >> x >> y;
        }
        board.exec_move(x, y);
        auto res = board.get_result();
        if (res.first) break;
    }
}

int main() {
    Game game;
    game.run();
}