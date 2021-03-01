import sys
sys.path.append('../build')
from library import MCTS, AlphaZero, Board, NeuralNetwork
sys.path.append('../src')
from neural_network import NeuralNetWorkWrapper

class Human():
    def __init__(self):
        pass

    def get_action(self, board):
        print("It's your turn. Input format: h w")
        while True:
            try:
                x, y = map(int, list(input().strip().split()))
                if not board.is_legal(x, y):
                    raise Exception
            except Exception:
                print("Invalid input. Try again.")
            else:
                break
        return x * board.get_n() + y

    def update_with_move(self, action):
        pass
                
class Game():
    def __init__(self):
        pass

    def contest(self, player1, player2, show = True):
        if isinstance(player1, Human) or isinstance(player2, Human):
            assert(show) # board must be shown when human play
        board = Board(8, 5)
        while True:
            if show:
                board.display()
            res = board.get_result()
            if res[0]:
                if show:
                    print("Game ended.\n")
                    if res[1] == 0:
                        print('Game ended in a draw.')
                    else:
                        print("Winner is the", "first" if res[1] == 1 else "second", "player")
                return res[1]
            if show:
                print("Now is", 'x' if board.get_cur_player() == 1 else 'o', "'s turn.\n")
            
            if board.get_cur_player() == 1:
                action = player1.get_action(board)
            else:
                action = player2.get_action(board)

            board.exec_move(action)
            player1.update_with_move(action)
            player2.update_with_move(action)

    def tournament(self, player1, player2, round = 10, show = True):
        win1_cnt, win2_cnt, draw_cnt = 0, 0, 0
        for i in range(round):
            if show:
                print("\nPlayer{:d} takes the first action.".format(1 if i < round // 2 else 2))
            if i < round // 2:
                res = self.contest(player1, player2, show)
                if res == 1:
                    win1_cnt += 1
                elif res == -1:
                    win2_cnt += 1
                else:
                    draw_cnt += 1
            else:
                res = self.contest(player2, player1, show)
                if res == 1:
                    win2_cnt += 1
                elif res == -1:
                    win1_cnt += 1
                else:
                    draw_cnt += 1
        print("\nTournament finished.")
        print("Player1 vs. player2: {:d}/{:d}/{:d} (win1/win2/draw)".format(win1_cnt, win2_cnt, draw_cnt))

network1 = NeuralNetwork("../models/model_8x8_5_2000iters_cpu.pt", False, 4)
network2 = NeuralNetwork("../models/model_8x8_5_2000iters_cpu.pt", False, 4)

game = Game()
human = Human()
mcts1 = MCTS(4, 100000, 5, 3)
mcts2 = MCTS(4, 100000, 5, 3)
alphazero1 = AlphaZero(network1, 4, 800, 5, 3)
alphazero2 = AlphaZero(network2, 4, 800, 5, 3)
game.contest(human, human)
# game.tournament(mcts1, mcts2, round = 10, show = True)