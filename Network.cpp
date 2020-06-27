#include "Network.h"

namespace Neuron {

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
    
    for (int layer = 0; layer < layers.size(); layer++) {
        layers.at(layer) = std::move(Matrix(1, dimensions.at(layer)));
    }

    for (int weight = 0; weight < weights.size(); weight++) {
        weights.at(weight) = RandomNormalMatrix(dimensions.at(weight), dimensions.at(weight + 1) , 0, 5);        
        bias.at(weight)    = std::move(Matrix      (1,                     dimensions.at(weight) + 1));
    }
}


Network::Network(const std::initializer_list<double> & list) : 
    dimensions (list.size()),
    layers     (std::vector<Matrix> (list.size())),
    weights    (std::vector<Matrix> (list.size() - 1)),
    bias       (std::vector<Matrix> (list.size() - 1))  {}


void Network::feedforward(const vector1D & inputs) {
    // output_layer = weights.at(last - 1).transpose multiply outputs of previous layer 

    for (auto & layer : layers) {
        if (layer == *layers.begin()) {
            layer = inputs;
        }
        else {
            layer = weights.at(i - 1).Transpose().dot(layers.at(i - 1)) + bias;
        }
    }
}



Matrix & Network::operator = (const std::initializer_list<double> & list) { 

    dimensions = std::vector<int>    (list.size());
    layers     = std::vector<Matrix> (list.size());
    weights    = std::vector<Matrix> (list.size() - 1);
    bias       = std::vector<Matrix> (list.size() - 1);  
    std::cout << "test" << std::endl;
}

void Network::train(vector2D inputs, vector2D output) {

}


vector1D Network::predict(vector1D inputs, vector1D output) const {

}


double Network::test(vector2D inputs, vector2D output) const {

}


std::vector<Matrix> Network::getWeights() const{
    return weights;
}


std::vector<Matrix> Network::getLayers() const{
    return layers;
}


std::vector<Matrix> Network::getBiasVector() const{
    return bias;
}

std::vector<int> Network::Dimensions() const {

}


std::pair<vector2D, vector2D> train_test_split(vector2D, const double & test_size) {

}

vector1D oneHot(vector1D output_layer) {

}

}