#include "Network.h"

namespace Neuron {

Network::Network() : 
    dimensions (std::vector<int>        (0)),
    layers     (std::vector<Matrix>     (0)),
    weights    (std::vector<Matrix>     (0)),
    bias       (std::vector<Matrix>     (0)) {}

Network::Network(const std::vector<int> & dimensions) : 
    dimensions (dimensions),
    layers     (std::vector<Matrix>       (dimensions.size()    )),
    layers_activated(std::vector<Matrix>  (dimensions.size()    )),
    weights    (std::vector<Matrix>       (dimensions.size() - 1)),
    activations(std::vector<Activation *> (dimensions.size() - 1)),
    bias       (std::vector<Matrix>       (dimensions.size() - 1)),
    learning_rate (0.01)  
{
    
    for (int layer = 0; layer < layers.size(); layer++) {
        layers.at(layer) = std::move(Matrix(dimensions.at(layer), 1));
        layers_activated.at(layer) = std::move(Matrix(dimensions.at(layer), 1));
    }

    for (int layer = 0; layer < layers.size() - 1; layer++) 
        activations.at(layer) = new ReLU();

    for (int weight = 0; weight < weights.size(); weight++) {
        weights.at(weight) = std::move(RandomNormalMatrix(dimensions.at(weight),     dimensions.at(weight + 1) , 0, 2));        
        bias.at(weight)    = std::move(RandomNormalMatrix(dimensions.at(weight + 1), 1                         , 0, 2));
    }
}


Network::Network(const std::initializer_list<double> & list) : 
    dimensions (list.size()),
    layers     (std::vector<Matrix> (list.size())),
    weights    (std::vector<Matrix> (list.size() - 1)),
    bias       (std::vector<Matrix> (list.size() - 1))  {}


void Network::feedforward(const Matrix & inputs) {   
    int i = 0;
    
    for (auto & layer : layers) {
        if (i == 0) {
            layer = inputs;
            std::cout << "first " << layer << std::endl;
        }
        else {
            std::cout << "weights transposed: " << weights.at(i - 1).Transpose() << std::endl;

            layer = weights.at(i - 1).Transpose().dot(layers.at(i - 1)) + bias.at(i - 1);
            std::cout << "i: " << i << std::endl;
            std::cout << "before activate" << std::endl << std::flush;
            layers_activated.at(i) = activations.at(i - 1)->activate(layer); 
            std::cout << "after activate" << std::endl << std::flush;

        }

        i++;
    }
}

void Network::backpropagate(const Matrix & correct_outputs) {
    Matrix error = Cost(getOutputLayer(), correct_outputs).multiply(activations.at(activations.size() - 1)->activate_prime(layers.at(layers.size() - 1)));

    for (int i = layers.size() - 2; i > 0; i--) {
        error = (weights.at(i + 1).Transpose().dot(error)).multiply(activations.at(i)->activate_prime(layers.at(i)));
    }

}

Matrix Network::getOutputLayer() const {
    return layers_activated.at(layers_activated.size() - 1);
}

Matrix & Network::operator = (const std::initializer_list<double> & list) { 

    dimensions = std::vector<int>    (list.size());
    layers     = std::vector<Matrix> (list.size());
    weights    = std::vector<Matrix> (list.size() - 1);
    bias       = std::vector<Matrix> (list.size() - 1);  
    std::cout << "test" << std::endl;
}

void Network::train(vector2D inputs, vector2D output) {
    feedforward(inputs);
}


vector1D Network::predict(vector1D inputs, vector1D output) const {

}


double Network::test(vector2D inputs, vector2D output) const {

}

vector2D Network::getOutput() const {
    std::cout << "output layer: " << getOutputLayer() << std::endl;
    return getOutputLayer().toVector();
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

Matrix Cost(const Matrix & outputs, const Matrix & correct_outputs) {
    if (outputs.shape() != correct_outputs.shape()) 
        throw MatrixDimensionError();
    
    Matrix output_matrix (outputs);

    for (unsigned int i = 0; i < outputs.size(); i++) 
        output_matrix(i) = outputs(i) - correct_outputs(i);

    return output_matrix;
}

Matrix squaredCost(const Matrix & outputs, const Matrix & correct_outputs) {
    if (outputs.shape() != correct_outputs.shape()) 
        throw MatrixDimensionError();
    
    Matrix output_matrix (outputs);

    for (unsigned int i = 0; i < outputs.size(); i++) 
        output_matrix(i) = 0.5 * std::pow(outputs(i) - correct_outputs(i), 2);

    return output_matrix;
}

std::pair<vector2D, vector2D> train_test_split(vector2D, const double & test_size) {

}

vector1D oneHot(vector1D output_layer) {

}

}