## AlphaZero_Gomoku

This is a multi-thread implementation of AlphaZero for Gomoku. 

MCTS (Monte Carlo Tree Search) is implemented in C++11 for efficiency (100,000 playouts with rollout policy only take a few seconds)  and neural network is implemented in Pytorch and Libtorch is used to run model in C++. Besides, SWIG is used to call C++ interfaces in Python while training models.

### Environment

AlphaZero_20210225.yml is my anaconda envrionment with

- Python 3.7
- Pytorch 1.2.0

Besides, CMake and SWIG is used:

- CMake 3.8.0
- SWIG 3.0.12

### Run

```
# compile
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # you may need to specify torch dir and python dir
cmake --build

# run
cd ../test
python game_test.py # to play with pure MCTS or model
python train_test.py # to train model
```

A simple 8x8 model trained on CPU with 2000 iterations is put in models/ . It has been fully trained and almost never lose in games.

### References
[alpha-zero-gomoku](https://github.com/hijkzzz/alpha-zero-gomoku) . A multi-threaded implementation of AlphaZero.

[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) . An implementation of the AlphaZero algorithm for Gomoku.

[ThreadPool](https://github.com/progschj/ThreadPool) . A simple C++11 Thread Pool implementation.

David Silver, Julian Schrittwieser, Karen Simonyan, et al. Mastering the game of Go without human knowledge.

Guillaume M.J-B. Chaslot, Mark H.M. Winands, and H. Jaap van den Herik. Parallel Monte-Carlo Tree Search.