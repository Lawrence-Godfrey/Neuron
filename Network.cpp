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
        } else {
            layer = weights.at(i - 1).Transpose().dot(layers.at(i - 1)) + bias.at(i - 1);
            layers_activated.at(i) = activations.at(i - 1)->activate(layer); 
        }

        i++;
    }
}

void Network::backpropagate(const Matrix & correct_outputs) {
    Matrix error = Cost(getOutputLayer(), correct_outputs).multiply(activations.back()->activate_prime(layers.back()));
    Matrix gradient = layers_activated.at(layers_activated.size() - 2).dot(error.Transpose());

    weights.back() = weights.back() - learning_rate * gradient;

    for (int i = layers.size() - 2; i > 0; i--) {
        error = (weights.at(i).dot(error)).multiply(activations.at(i)->activate_prime(layers.at(i)));
        
        gradient = layers_activated.at(i - 1).dot(error.Transpose());
                
        weights.at(i - 1) = weights.at(i - 1) - learning_rate * gradient;
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
}

void Network::train (const std::vector<Matrix> & inputs, const std::vector<Matrix> & outputs) {
    int i = 0;
    
    for (int i = 0; i < inputs.size(); i++) {
        if(100 * (double(i)/double(inputs.size())) > 2)
            break;

        std::cout << "Progress: " <<std::setprecision(2) <<100 * (double(i)/double(inputs.size())) << "%" << std::flush;
        feedforward(inputs.at(i));
        backpropagate(outputs.at(i));
        printf("\033[1K");
        printf("\033[0E");
    }
        
}

void Network::train(vector2D inputs, vector2D output) {
    feedforward(inputs);
}


vector1D Network::predict(vector1D inputs, vector1D output) const {
    return output;
}


double Network::test(vector2D inputs, vector2D output) const {
    return 0;
}

vector2D Network::getOutput() const {
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
    return dimensions;
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
    return output_layer;
}

}