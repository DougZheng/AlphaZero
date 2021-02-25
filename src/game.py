import sys
sys.path.append('../build')
from library import NeuralNetwork
from library import MCTS, AlphaZero
from library import Board
# from neural_network import NeuralNetworkWrapper

class Game():
    def __init__(self):
        pass
    def run(self):
        board = Board(9, 5)
        player1 = MCTS(4, 5000, 5, 3)
        # player2 = MCTS(4, 1500, 5, 3)
        neural_network = NeuralNetwork("../models/best_checkpoint.pt", False, 4)
        player2 = AlphaZero(neural_network, 4, 800, 5, 3)
        while True:
            board.display()
            res = board.get_result()
            if res[0]:
                break
            print("now for", 'x' if board.get_cur_player() == 1 else 'o')
            if board.get_cur_player() == 1:
                print()
                action = player1.get_action(board)
                board.exec_move(action)
                player1.update_with_move(action)
                player2.update_with_move(action)
            else:
                print()
                action = player2.get_action(board)
                board.exec_move(action)
                player1.update_with_move(action)
                player2.update_with_move(action)
        print("finish")

game = Game()
game.run()