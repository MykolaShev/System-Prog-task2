#include <iostream>
#include <vector>
#include <random>
#include <chrono>

const int N = 1024;

void initMatrix(std::vector<float>& mat) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dist(gen);
    }
}

/**
 * Performs naive O(N^3) matrix multiplication: C = A * B.
 * Uses standard row-by-column dot product computation.
 * @param A Input matrix A (N x N).
 * @param B Input matrix B (N x N).
 * @param C Output matrix C (N x N).
 */
void multiplyMatrix(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                // Accessing B[k * N + j] causes frequent cache misses
                // because we read columns, not rows.
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    initMatrix(A);
    initMatrix(B);

    auto start = std::chrono::high_resolution_clock::now();
    
    multiplyMatrix(A, B, C);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "Matrix multiplication completed.\n";
    std::cout << "Execution time: " << diff.count() << " seconds\n";
    
    std::cout << "Verification check (C[0]): " << C[0] << "\n";

    return 0;
}