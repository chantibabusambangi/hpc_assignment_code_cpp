#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono> // For measuring execution time

// Function to compute the hypothesis (prediction)
double hypothesis(const std::vector<double>& theta, const std::vector<double>& x) {
    double h = 0.0;
    for (size_t i = 0; i < theta.size(); ++i) {
        h += theta[i] * x[i];
    }
    return h;
}

// Batch Gradient Descent (Serial)
void batchGradientDescent(std::vector<double>& theta, const std::vector<std::vector<double>>& X, const std::vector<double>& y, double alpha, int iterations) {
    int m = X.size(); // Number of training examples
    int n = X[0].size(); // Number of features

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> gradient(n, 0.0);

        // Compute gradients
        for (int i = 0; i < m; ++i) {
            double error = hypothesis(theta, X[i]) - y[i];
            for (int j = 0; j < n; ++j) {
                gradient[j] += error * X[i][j];
            }
        }

        // Update theta
        for (int j = 0; j < n; ++j) {
            theta[j] -= (alpha / m) * gradient[j];
        }

        // Print progress every 1000 iterations
        if (iter % 1000 == 0) {
            std::cout << "Iteration: " << iter << "/" << iterations << " completed.\n";
        }
    }
}

int main() {
    // Increase dataset size for more execution time
    int num_samples = 5000; // 5 million samples
    int num_features = 100;    // 100 features

    std::vector<std::vector<double>> X(num_samples, std::vector<double>(num_features, 1.0));
    std::vector<double> y(num_samples, 0.0);
    
    // Initialize random data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 1; j < num_features; ++j) { // Start from index 1 (bias term at index 0)
            X[i][j] = dist(gen); // Random feature values
        }        y[i] = 1.5 * X[i][1] + 2.2 * X[i][2] - 0.8 * X[i][3] + 3.7 * X[i][4] + dist(gen); // Linear function + noise
    }

    // Initialize theta
    std::vector<double> theta(num_features, 0.0);

    // Hyperparameters
    double alpha = 0.0001; // Learning rate (smaller for large dataset)
    int iterations = 10000; // Increased iterations for more computation

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run serial Batch Gradient Descent
    batchGradientDescent(theta, X, y, alpha, iterations);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    
    // Output the learned parameters
    std::cout << "Theta: ";
    for (double t : theta) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    std::cout << "Execution Time: " << elapsed_time.count() << " seconds\n";

    return 0;
}
