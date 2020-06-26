#pragma once

#include "../Matrix/Matrix.h"
#include <iostream>
#include <vector>
#include <initializer_list>

using vector2D = std::vector<std::vector<double>>; 
using vector1D = std::vector<double>;

class Network {
    std::vector<int> dimensions;

    std::vector<Matrix> layers;
    std::vector<Matrix> weights;
    std::vector<Matrix> bias;

    vector1D feedforward(const vector1D & inputs);
    void backpropagate();

    public:
        Network();
        Network(const std::vector<int> & dimensions);
        Network(const std::initializer_list<double> & list);

        Matrix & operator = (const std::initializer_list<double> & list);

        void train(vector2D inputs, vector2D output);
        
        vector1D predict(vector1D inputs, vector1D output) const;
        
        double test(vector2D inputs, vector2D output) const;
        
        template <typename T>
        void addLayer(const int & layer_size);

        std::vector<int> Dimensions() const;
        
};

template<typename T>
void Network::addLayer(const int & layer_size) {
    
}

std::pair<vector2D, vector2D> train_test_split(vector2D, const double & test_size);

vector1D oneHot(vector1D output_layer);
