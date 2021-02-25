from collections import deque
from os import path, mkdir
import threading
import time
import math
import numpy as np
import pickle
import concurrent.futures
import random
from functools import reduce

import sys
sys.path.append('../build')
from library import MCTS, AlphaZero, Board, NeuralNetwork

from neural_network import NeuralNetWorkWrapper
import logging


def tuple2d_to_numpy2d(tuple2d):
    res = [None] * len(tuple2d)
    for i, tuple1d in enumerate(tuple2d):
        res[i] = list(tuple1d)
    return np.array(res)

class TrainPipeline():
    def __init__(self, config):

        logging.basicConfig(filename = config['train_log_file'], 
            level = logging.DEBUG, format = '%(message)s', filemode = 'w')

        self.n = config['n']
        self.n_in_row = config['n_in_row']
        self.action_size = config['action_size']

        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer = deque([], maxlen = config['examples_buffer_max_len'])

        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'], 
            config['num_channels'], config['n'], config['action_size'], config['train_use_gpu'], config['libtorch_use_gpu'])
        
        self.show_train_board = config['show_train_board']

    def learn(self):
        if path.exists(path.join('models', 'checkpoint.example')):
            logging.debug('loading checkpoint...')
            self.nnet.load_model()
            self.load_samples() 
        else:
            self.nnet.save_model()
            self.nnet.save_model('models', 'best_checkpoint')
        
        for itr in range(1, self.num_iters + 1):
            logging.debug('-' * 65)
            logging.debug('iter: {}'.format(itr))
            logging.debug('-' * 65)
            libtorch = NeuralNetwork('./models/checkpoint.pt', self.libtorch_use_gpu, self.num_mcts_sims * self.num_train_threads)
            itr_examples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers = self.num_train_threads) as executor:
                futures = [executor.submit(self.self_play, 1 if itr % 2 else -1, 
                    libtorch, self.show_train_board if k == 1 else False) for k in range(1, self.num_eps + 1)]
                for k, f in enumerate(futures):
                    examples = f.result()
                    itr_examples += examples
                    remain = min(len(futures) - (k + 1), self.num_train_threads)
                    libtorch.set_batch_size(max(remain * self.num_mcts_threads, 1))
                    logging.debug('eps: {}, examples: {}, moves: {}'.format(k + 1, len(examples), len(examples) // 8) )
            del libtorch
            self.examples_buffer.append(itr_examples)
            train_data = reduce(lambda a, b : a + b, self.examples_buffer)
            random.shuffle(train_data)
            epochs = self.epochs * (len(itr_examples) + self.batch_size - 1) // self.batch_size
            epoch_res = self.nnet.train(train_data, self.batch_size, int(epochs))
            for epo, loss, entropy in epoch_res:
                logging.debug("epoch: {}, loss: {}, entropy: {}".format(epo, loss, entropy))
            self.nnet.save_model()
            self.save_samples()
            
            if itr % self.check_freq == 0:
                num_half_threads = max(self.num_mcts_threads * self.num_train_threads // 2, 1)
                libtorch_current = NeuralNetwork('./models/checkpoint.pt', self.libtorch_use_gpu, num_half_threads)
                libtorch_best = NeuralNetwork('./models/best_checkpoint.pt', self.libtorch_use_gpu, num_half_threads)
                win_cnt, lose_cnt, draw_cnt = self.contest(libtorch_current, libtorch_best, self.num_contest)
                logging.debug('new vs. prev: {:d} wins, {:d} loses, {:d} draws'.format(win_cnt, lose_cnt, draw_cnt))
                if win_cnt + lose_cnt > 0 and win_cnt / (win_cnt + lose_cnt) > self.update_threshold:
                    logging.debug('new model accepted.')
                    self.nnet.save_model('models', 'best_checkpoint')
                else:
                    logging.debug('new model rejected')
                del libtorch_current
                del libtorch_best
    
    def self_play(self, start_player, libtorch, show):
        if show:
            print('display of a self play round begins\n')

        train_examples = []
        player1 = AlphaZero(libtorch, self.num_mcts_threads, self.num_mcts_sims, self.c_puct, self.c_virtual_loss)
        player2 = AlphaZero(libtorch, self.num_mcts_threads, self.num_mcts_sims, self.c_puct, self.c_virtual_loss)
        players = [player2, None, player1]
        player_index = 1
        board = Board(self.n, self.n_in_row, start_player)

        episode_step = 0
        while True:
            episode_step += 1
            player = players[player_index + 1]
            if episode_step <= self.num_explore:
                prob = np.array(list(player.get_action_probs(board, self.temp)))
            else:
                prob = np.array(list(player.get_action_probs(board, 0)))
            board_states = tuple2d_to_numpy2d(board.get_states())
            last_action = board.get_last_move()
            cur_player = board.get_cur_player()

            sym = self.get_symmetries(board_states, prob, last_action)
            for b, p, a in sym:
                train_examples.append([b, a, cur_player, p])

            legal_moves = list(board.get_moves())
            noise = 0.1 * np.random.dirichlet(self.dirichlet_alpha * np.ones(len(legal_moves)))

            prob = 0.9 * prob
            for i in range(len(legal_moves)):
                prob[legal_moves[i]] += noise[i]
            prob /= np.sum(prob)

            action = np.random.choice(len(prob), p = prob)
                
            board.exec_move(action)
            if show:
                board.display()
            player1.update_with_move(action)
            player2.update_with_move(action)

            player_index = -player_index

            ended, winner = board.get_result()
            if ended:
                break
        if show:
            print('display of a self play round finished\n')

        return [(x[0], x[1], x[2], x[3], x[2] * winner) for x in train_examples]

    def contest(self, network1, network2, num_contest):
        win_cnt, lose_cnt, draw_cnt = 0, 0, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers = self.num_train_threads) as executor:
            futures = [executor.submit(self._contest, network1, network2, 
                1 if k <= num_contest // 2 else -1, self.show_train_board if k == 1 else 0) for k in range(1, num_contest + 1)]
            for f in futures:
                winner = f.result()
                if winner == 1:
                    win_cnt += 1
                elif winner == -1:
                    lose_cnt += 1
                else:
                    draw_cnt += 1
        return win_cnt, lose_cnt, draw_cnt

    def _contest(self, network1, network2, start_player, show):
        if show:
            print('display of a contest round begins\n')

        player1 = AlphaZero(network1, self.num_mcts_threads, self.num_mcts_sims, self.c_puct, self.c_virtual_loss)
        player2 = AlphaZero(network2, self.num_mcts_threads, self.num_mcts_sims, self.c_puct, self.c_virtual_loss)
        players = [player2, None, player1]
        player_index = start_player
        board = Board(self.n, self.n_in_row, start_player)

        while True:
            player = players[player_index + 1]
            best_move = player.get_action(board)
            board.exec_move(best_move)

            if show:
                board.display()
            
            ended, winner = board.get_result()
            if ended == 1:
                if show:
                    print('display of a contest round begins\n')
                return winner
            
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)
            player_index = -player_index

    def get_symmetries(self, board, pi, last_action):
        assert(len(pi) == self.action_size)
        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        last_action_board[last_action // self.n][last_action % self.n] = 1
        res = []
        for i in range(4):
            newB = np.rot90(board, i)
            newPi = np.rot90(pi_board, i)
            newAction = np.rot90(last_action_board, i)
            res += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
            newB = np.fliplr(newB)
            newPi = np.fliplr(newPi)
            newAction = np.fliplr(newAction)
            res += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return res

    def load_samples(self, folder = 'models', filename = 'checkpoint.example'):
        filepath = path.join(folder, filename)
        with open(filepath, 'rb') as f:
            self.examples_buffer = pickle.load(f)
        
    def save_samples(self, folder = 'models', filename = 'checkpoint.example'):
        if not path.exists(folder):
            mkdir(folder)
        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.examples_buffer, f, -1)