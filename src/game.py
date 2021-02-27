import sys
sys.path.append('../build')
from library import MCTS, AlphaZero, Board, NeuralNetwork
from neural_network import NeuralNetWorkWrapper

class Game():
    def __init__(self):
        pass
    def run(self):
        board = Board(8, 5)
        player1 = MCTS(4, 800, 5, 3)
        # player2 = MCTS(4, 1800, 5, 3)
        neural_network = NeuralNetwork("../src/best_checkpoint.pt", False, 4)
        player2 = AlphaZero(neural_network, 4, 800, 5, 3)
        while True:
            board.display()
            res = board.get_result()
            if res[0]:
                break
            print("now for", 'x' if board.get_cur_player() == 1 else 'o')
            print()
            if board.get_cur_player() == 1:
                action = player1.get_action(board)
                board.exec_move(action)
            else:
                action = player2.get_action(board)
                board.exec_move(action)
            
            player1.update_with_move(action)
            player2.update_with_move(action)
        print("finish")

game = Game()
game.run()