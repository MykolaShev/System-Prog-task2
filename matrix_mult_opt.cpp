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
 * Performs cache-friendly O(N^3) matrix multiplication: C = A * B.
 * Loop order is swapped from i-j-k to i-k-j to ensure sequential memory access (row-major).
 * This layout eliminates L1 cache misses and allows the compiler to apply SIMD (AVX) vectorization.
 * 
 * @param A Input matrix A (N x N).
 * @param B Input matrix B (N x N).
 * @param C Output matrix C (N x N).
 */
void multiplyMatrix(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // Cache the scalar value to avoid repeated memory lookups
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                // Sequential memory access for both C and B
                C[i * N + j] += a_ik * B[k * N + j];
            }
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