#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

@interface MatrixMultiplication : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> defaultLibrary;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineState;

- (instancetype)init;
- (void)runBatchMatrixMultiplicationWithBatchSize:(NSUInteger)batchSize
                                         matrixAWidth:(NSUInteger)widthA
                                         matrixBWidth:(NSUInteger)widthB;

@end

@implementation MatrixMultiplication

- (instancetype)init {
    if (self = [super init]) {
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
        NSError *error = nil;
        NSString *path = [[NSBundle mainBundle] pathForResource:@"matrix_multiplication" ofType:@"metallib"];
        id<MTLLibrary> library = [_device newLibraryWithFile:path error:&error];

        if (error) {
            NSLog(@"Failed to load Metal library: %@", error);
            return 0;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"matrixMultiply"];
        if (!kernelFunction) {
            NSLog(@"Failed to find function 'matrixMultiply' in Metal library.");
            return 0;
        }
        _pipelineState = [_device newComputePipelineStateWithFunction:kernelFunction error:nil];
    }
    return self;
}

- (float *)generateRandomMatrixWithRows:(NSUInteger)rows cols:(NSUInteger)cols {
    NSUInteger size = rows * cols;
    float *matrix = (float *)malloc(size * sizeof(float));
    for (NSUInteger i = 0; i < size; i++) {
        matrix[i] = (float)arc4random_uniform(10);  // Random number between 0 and 9
    }
    return matrix;
}

- (void)runBatchMatrixMultiplicationWithBatchSize:(NSUInteger)batchSize
                                         matrixAWidth:(NSUInteger)widthA
                                         matrixBWidth:(NSUInteger)widthB {

    
    NSUInteger matrixAHeight = widthA;  // Assuming square matrices for simplicity

    // Prepare random matrices for A, B, and allocate C
    float *matrixA = [self generateRandomMatrixWithRows:matrixAHeight cols:widthA * batchSize];
    float *matrixB = [self generateRandomMatrixWithRows:widthA cols:widthB * batchSize];
    float *matrixC = (float *)malloc(matrixAHeight * widthB * batchSize * sizeof(float));

    uint64_t startTotalTime = mach_absolute_time();

    // Create buffers
    id<MTLBuffer> bufferA = [_device newBufferWithBytes:matrixA length:matrixAHeight * widthA * batchSize * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [_device newBufferWithBytes:matrixB length:widthA * widthB * batchSize * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [_device newBufferWithLength:matrixAHeight * widthB * batchSize * sizeof(float) options:MTLResourceStorageModeShared];
    
    uint64_t startSetTime = mach_absolute_time();
    // Create command buffer and compute command encoder
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:_pipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];

    // Set constants (matrix dimensions)
    uint widthA_const = (uint)widthA;
    uint widthB_const = (uint)widthB;
    uint batchSize_const = (uint)batchSize;
    [encoder setBytes:&widthA_const length:sizeof(uint) atIndex:3];
    [encoder setBytes:&widthB_const length:sizeof(uint) atIndex:4];
    [encoder setBytes:&batchSize_const length:sizeof(uint) atIndex:5];

    // Define grid and thread size
    MTLSize gridSize = MTLSizeMake(widthB, matrixAHeight * batchSize, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);  // 16x16 threads per threadgroup
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];

    uint64_t startComputeTime = mach_absolute_time();
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    uint64_t endTotalTime = mach_absolute_time();
    
    // total time
    uint64_t elapsedTotalTime = endTotalTime - startTotalTime;
    uint64_t elapsedSetTime = startComputeTime - startSetTime;
    uint64_t elapsedIssueTime = startSetTime - startTotalTime;
    uint64_t elapsedComputeTime = endTotalTime - startComputeTime;
    mach_timebase_info_data_t timebaseInfo;
    mach_timebase_info(&timebaseInfo);
    double elapsedTotalTimeInNanoseconds = (double)elapsedTotalTime * (double)timebaseInfo.numer / (double)timebaseInfo.denom;
    double elapsedIssueTimeInNanoseconds = (double)elapsedIssueTime * (double)timebaseInfo.numer / (double)timebaseInfo.denom;
    double elapsedSetTimeInNanoseconds = (double)elapsedSetTime * (double)timebaseInfo.numer / (double)timebaseInfo.denom;
    double elapsedComputeTimeInNanoseconds = (double)elapsedComputeTime * (double)timebaseInfo.numer / (double)timebaseInfo.denom;
    printf("%lu, %lu, %.3f, %.3f, %.3f, %.3f\n", batchSize, widthA, elapsedTotalTimeInNanoseconds / 1e6, elapsedIssueTimeInNanoseconds / 1e6, elapsedSetTimeInNanoseconds / 1e6, elapsedComputeTimeInNanoseconds / 1e6);   //output milliseconds
        

    // Retrieve results
    //memcpy(matrixC, bufferC.contents, matrixAHeight * widthB * batchSize * sizeof(float));
    
    // Output result (example)
    /***
    NSLog(@"Result matrix C:");
    for (NSUInteger b = 0; b < batchSize; b++) {
        NSLog(@"Batch %lu:", (unsigned long)b);
        for (NSUInteger i = 0; i < matrixAHeight; i++) {
            NSMutableString *rowString = [NSMutableString string];
            for (NSUInteger j = 0; j < widthB; j++) {
                [rowString appendFormat:@"%f ", matrixC[b * matrixAHeight * widthB + i * widthB + j]];
            }
            NSLog(@"%@", rowString);
        }
    }
    *///

    // Clean up
    free(matrixA);
    free(matrixB);
    free(matrixC);
}

@end

int main(int argc, const char * argv[]) {
    NSUInteger batch_sizes[] = {1, 2, 4, 8, 16, 32};
    NSUInteger shapes[] = {1000, 2000, 3000, 4000};
    //NSArray *batch_sizes = @[@1, @2, @4, @8, @16, @32];
    //NSArray *shapes = @[@1000, @2000, @3000, @4000];
    printf("bs, shape, total, issue, setup, computation\n");
    for (int i=0; i<6; i++) {
        for (int j=0; j<4; j++){
            @autoreleasepool {
                MatrixMultiplication *matrixMultiplication = [[MatrixMultiplication alloc] init];
                [matrixMultiplication runBatchMatrixMultiplicationWithBatchSize:batch_sizes[i] matrixAWidth:shapes[j] matrixBWidth:shapes[j]];
            }
        }
    }
    return 0;
}
