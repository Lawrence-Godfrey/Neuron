#include "Network.h"

Network::Network() : 
    dimensions (std::vector<int>    (0)),
    layers     (std::vector<Matrix> (0)),
    weights    (std::vector<Matrix> (0)),
    bias       (std::vector<Matrix> (0)) {}

Network::Network(const std::vector<int> & dimensions) : 
    dimensions (dimensions),
    layers     (std::vector<Matrix> (dimensions.size())),
    weights    (std::vector<Matrix> (dimensions.size() - 1)),
    bias       (std::vector<Matrix> (dimensions.size() - 1))  {
    
    for (int layer = 0, weight = 0; layer < dimensions.size(), weight < dimensions.size() - 1; layer++, weight++) {
        layers.at(layer) = Matrix(1, dimensions.at(layer));
        weights.at(weight) = RandomNormalMatrix(dimensions.at(weight + 1), dimensions.at(weight));
        bias.at(weight) = Matrix(1, dimensions.at(weight));
    }


}




Network::Network(const std::initializer_list<double> & list) : 
    dimensions (list.size()),
    layers     (std::vector<Matrix> (list.size())),
    weights    (std::vector<Matrix> (list.size() - 1)),
    bias       (std::vector<Matrix> (list.size() - 1))  {}

Matrix & Network::operator = (const std::initializer_list<double> & list) { 

    dimensions = std::vector<int>    (list.size());
    layers     = std::vector<Matrix> (list.size());
    weights    = std::vector<Matrix> (list.size() - 1);
    bias       = std::vector<Matrix> (list.size() - 1);  

}

void Network::train(vector2D inputs, vector2D output) {

}


vector1D Network::predict(vector1D inputs, vector1D output) const {

}


double Network::test(vector2D inputs, vector2D output) const {

}


std::vector<int> Network::Dimensions() const {

}


std::pair<vector2D, vector2D> train_test_split(vector2D, const double & test_size) {

}

vector1D oneHot(vector1D output_layer) {

}