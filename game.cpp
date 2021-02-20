#include "game.h"
#include "mcts.h"

Game::Game() {

}

void Game::run() {
    while (true) {
        board.display();
        auto res = board.get_result();
        if (res.first) break;
        std::cout << "now for " << (board.get_cur_player() == 1 ? 'x' : 'o') << ": ";
        if (board.get_cur_player() == 1) {
            // int x, y;
            // std::cin >> x >> y;
            // if (!board.is_legal(x, y)) {
            //     std::cout << "not legal" << std::endl;
            //     continue;
            // }
            // board.exec_move(x, y);
            std::cout << std::endl;
            MCTS ai(10000);
            int pos = ai.get_move(board);
            board.exec_move(pos);
        }
        else {
            std::cout << std::endl;
            MCTS ai(10000);
            int pos = ai.get_move(board);
            board.exec_move(pos);
        }
    }
}

int main() {
    Game game;
    game.run();
}