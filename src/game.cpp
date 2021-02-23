#include "game.h"
#include "mcts.h"

Game::Game() {

}
void Game::run() {
    NeuralNetwork *neural_network = new NeuralNetwork("../models/best_checkpoint.pt", false, 4);
    MCTS player1(4, 15000, 5, 3);
    AlphaZero player2(neural_network, 4, 1500, 5, 3);
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
            int action = player1.get_action(board);
            board.exec_move(action);
            player1.update_with_move(action);
            player2.update_with_move(action);
        }
        else {
            std::cout << std::endl;
            int action = player2.get_action(board);
            board.exec_move(action);
            player1.update_with_move(action);
            player2.update_with_move(action);
        }
    }
    delete neural_network;
}

int main() {
    Game game;
    game.run();
}