%module library

%{
#include "mcts.h"
#include "board.h"
#include "neural_network.h"
%}

%include "std_vector.i"
namespace std {
    %template(IntVector) vector<int>;
    %template(IntVectorVector) vector<vector<int>>;
    %template(DoubleVector) vector<double>;
    %template(DoubleVectorVector) vector<vector<double>>;
    %template(IntVectorVectorVector) vector<vector<vector<int>>>;
}

%include "std_pair.i"
namespace std {
    %template(BoolIntPair) pair<bool, int>;
    %template(DoubleVectorDoublePair) pair<vector<double>, double>;
}

%include "std_string.i"

%include "mcts.h"
%include "board.h"

class NeuralNetwork {
public:
    NeuralNetwork(std::string model_path, bool use_gpu, unsigned batch_size);
    ~NeuralNetwork();
    void set_batch_size(unsigned batch_size);
};