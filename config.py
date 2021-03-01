config = {
    # board config
    'n': 11,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts config
    'libtorch_use_gpu' : False,                 # libtorch use cuda
    'num_mcts_threads': 4,                      # mcts threads number
    'num_mcts_sims': 1600,                      # mcts simulation times
    'c_puct': 5,                                # puct coeff
    'c_virtual_loss': 3,                        # virtual loss coeff

    # neural_network config
    'train_use_gpu' : False,                    # train neural network using cuda
    'lr': 0.001,                                # learning rate
    'l2': 0.0001,                               # L2 regular coeff
    'num_channels': 256,                        # convolution neural network channel size
    'num_layers' : 4,                           # residual layer number
    'epochs': 1.5,                              # train epochs
    'batch_size': 512,                          # batch size

    # train config
    'num_iters': 10000,                         # train iterations
    'num_eps': 4,                               # self play times in per iter
    'num_train_threads': 4,                     # self play in parallel
    'num_explore': 20,                          # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.3,                     # action noise in self play games
    'update_threshold': 0.55,                   # update model threshold
    'num_contest': 10,                          # new/old model compare times
    'check_freq': 20,                           # test model frequency
    'examples_buffer_max_len': 20,              # max length of examples buffer

    # train debug config
    'show_train_board' : True,                  # show action probs and board states
    'train_log_file' : '../train.log'           # logging
}

# action size
config['action_size'] = config['n'] ** 2