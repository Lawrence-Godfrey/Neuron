#pragma once

#include "../Matrix/Matrix.h"
#include <iostream>
#include <vector>
#include <initializer_list>

namespace Neuron {

using vector2D = std::vector<std::vector<double>>; 
using vector1D = std::vector<double>;

class Activation;
class Step;
class Sigmoid;
class ReLU;

class Network {
    
    std::vector<int>        dimensions;

    std::vector<Matrix>       layers;
    std::vector<Matrix>       layers_activated;
    std::vector<Matrix>       weights;
    std::vector<Matrix>       bias;

    std::vector<Activation *> activations;

    double learning_rate;
    
    Matrix getOutputLayer() const;
   
    public:
    void backpropagate(const Matrix & correct_outputs);

    void feedforward(const Matrix & inputs);
        Network();
        Network(const std::vector<int> & dimensions);
        Network(const std::initializer_list<double> & list);

        Matrix & operator = (const std::initializer_list<double> & list);

        void     train  (vector2D inputs, vector2D output); 
        void     train  (const std::vector<Matrix> & inputs, const std::vector<Matrix> & outputs);
        vector1D predict(vector1D inputs, vector1D output) const;    
        double   test   (vector2D inputs, vector2D output) const;
        
        template <typename T>
        void addLayer(const int & layer_size);

        template <typename T>
        void setLayer(const int & layer);

        vector2D getOutput() const;

        std::vector<Matrix> getWeights()    const;
        std::vector<Matrix> getLayers()     const;
        std::vector<Matrix> getBiasVector() const;

        std::vector<int> Dimensions() const;
        
};

Matrix Cost(const Matrix & outputs, const Matrix & correct_outputs);
Matrix squaredCost(const Matrix & outputs, const Matrix & correct_outputs);

template<typename T>
void Network::addLayer(const int & layer_size) {
    
}

std::pair<vector2D, vector2D> train_test_split(vector2D, const double & test_size);

vector1D oneHot(vector1D output_layer);

class Activation {
    public:
        virtual Matrix activate(const Matrix & layer) {
            return layer;
        }

        virtual Matrix activate_prime(const Matrix & layer) {
            return layer;
        }
};

class Step : public Activation {
    public:
        
        virtual Matrix activate(const Matrix & layer) override {
            Matrix output_matrix (layer);

            for (unsigned int i = 0; i < layer.size(); i++) 
                output_matrix(i) = int(output_matrix(i) >= 0.5);

            return output_matrix;
        }

        virtual Matrix activate_prime(const Matrix & layer) override {
            return activate(layer);
        }

};

class Sigmoid : public Activation {
    public:
        virtual Matrix activate(const Matrix & layer) override {
            Matrix output_matrix (layer);

            for (unsigned int i = 0; i < layer.size(); i++) 
                output_matrix(i) = 1 / (1 + std::exp(-1 * output_matrix(i)));

            return output_matrix;
        }

        virtual Matrix activate_prime(const Matrix & layer) override {
            return activate(layer).multiply(1 - activate(layer));
        }

};

class ReLU : public Activation {
    public: 
        virtual Matrix activate(const Matrix & layer) override {
            Matrix output_matrix (layer);
            for (unsigned int i = 0; i < layer.size(); i++) 
                output_matrix(i) = std::max(0.0, output_matrix(i));

            return output_matrix;
        }

        virtual Matrix activate_prime(const Matrix & layer) override {
            Matrix output_matrix (layer);

            for (unsigned int i = 0; i < layer.size(); i++) 
                output_matrix(i) = int(output_matrix(i) >= 0.5);

            return output_matrix;
        }

};

}