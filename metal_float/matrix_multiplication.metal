#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiply(
    device const float* A [[ buffer(0) ]],
    device const float* B [[ buffer(1) ]],
    device float* C [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]],
    constant uint &widthA [[ buffer(3) ]],
    constant uint &widthB [[ buffer(4) ]],
    constant uint &batchSize [[ buffer(5) ]]
) {
    uint batchIndex = gid.y / widthA;  // Calculate which batch we're in
    uint row = gid.y % widthA;         // Row of matrix A
    uint col = gid.x;                  // Column of matrix B

    float sum = 0.0;
    for (uint k = 0; k < widthA; k++) {
        sum += A[batchIndex * widthA * widthA + row * widthA + k] * B[batchIndex * widthA * widthB + k * widthB + col];
    }

    //C[batchIndex * widthA * widthB + row * widthB + col] = sum;
}
