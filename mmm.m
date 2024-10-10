#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

@interface MatrixMultiplication : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineState;

- (instancetype)init;
- (NSArray<NSNumber *> *)randomMatrixWithRows:(NSUInteger)rows columns:(NSUInteger)columns;
- (void)multiplyMatricesWithBatchSize:(NSUInteger)batchSize M:(NSUInteger)M N:(NSUInteger)N K:(NSUInteger)K;

@end

@implementation MatrixMultiplication

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
        
        // 加载Metal函数
        NSError *error = nil;
        NSString *path = [[NSBundle mainBundle] pathForResource:@"matrix_multiplication" ofType:@"metallib"];
        id<MTLLibrary> library = [_device newLibraryWithFile:path error:&error];

        if (error) {
            NSLog(@"Failed to load Metal library: %@", error);
            return 0;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"matrix_multiplication"];
        if (!kernelFunction) {
            NSLog(@"Failed to find function 'matrix_multiplication' in Metal library.");
            return 0;
        }
        _pipelineState = [_device newComputePipelineStateWithFunction:kernelFunction error:nil];
    }
    return self;
}

- (NSArray<NSNumber *> *)randomMatrixWithRows:(NSUInteger)rows columns:(NSUInteger)columns {
    NSMutableArray<NSNumber *> *matrix = [NSMutableArray arrayWithCapacity:rows * columns];
    for (NSUInteger i = 0; i < rows * columns; i++) {
        [matrix addObject:@(arc4random_uniform(100) / 100.0)];
    }
    return matrix;
}

// Helper function to convert NSArray<NSNumber *> to C array
- (float *)convertToCArray:(NSArray<NSNumber *> *)array {
    NSUInteger count = array.count;
    float *cArray = malloc(sizeof(float) * count);
    for (NSUInteger i = 0; i < count; i++) {
        cArray[i] = array[i].floatValue;
    }
    return cArray;
}

- (void)multiplyMatricesWithBatchSize:(NSUInteger)batchSize M:(NSUInteger)M N:(NSUInteger)N K:(NSUInteger)K {
    
    for (NSUInteger batch = 0; batch < batchSize; batch++) {
        NSArray<NSNumber *> *matrixA = [self randomMatrixWithRows:M columns:N];
        NSArray<NSNumber *> *matrixB = [self randomMatrixWithRows:N columns:K];
        NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:M * K];
        for (NSUInteger i = 0; i < M * K; i++) {
            [result addObject:@(0)];
        }
        
        // Convert NSArray<NSNumber *> to C arrays
        float *matrixACArray = [self convertToCArray:matrixA];
        float *matrixBCArray = [self convertToCArray:matrixB];
        float *resultCArray = calloc(M * K, sizeof(float));
        
        id<MTLBuffer> bufferA = [_device newBufferWithBytes:matrixACArray length:sizeof(float) * M * N options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [_device newBufferWithBytes:matrixBCArray length:sizeof(float) * N * K options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResult = [_device newBufferWithBytes:resultCArray length:sizeof(float) * M * K options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_pipelineState];
        [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferResult offset:0 atIndex:2];
        
        uint32_t m = (uint32_t)M;
        uint32_t n = (uint32_t)N;
        uint32_t k = (uint32_t)K;
        [computeEncoder setBytes:&m length:sizeof(m) atIndex:3];
        [computeEncoder setBytes:&n length:sizeof(n) atIndex:4];
        [computeEncoder setBytes:&k length:sizeof(k) atIndex:5];
        
        MTLSize gridSize = MTLSizeMake(K, M, 1);
        NSUInteger threadGroupSize = MIN(_pipelineState.maxTotalThreadsPerThreadgroup, K);
        MTLSize threadGroup = MTLSizeMake(threadGroupSize, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroup];
        uint64_t startTotalTime = mach_absolute_time();
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        uint64_t endTotalTime = mach_absolute_time();
    
        // 计算总时间
        uint64_t elapsedTime = endTotalTime - startTotalTime;
        mach_timebase_info_data_t timebaseInfo;
        mach_timebase_info(&timebaseInfo);
        double elapsedTimeInNanoseconds = (double)elapsedTime * (double)timebaseInfo.numer / (double)timebaseInfo.denom;
        printf("Total time for all batches: %.3f milliseconds\n", elapsedTimeInNanoseconds / 1e6);
        
        
        float *resultPointer = bufferResult.contents;
        //for (NSUInteger i = 0; i < M; i++) {
            //for (NSUInteger j = 0; j < K; j++) {
                //printf("%.2f ", resultPointer[i * K + j]);
            ///}
            //printf("\n");
        //}
        //printf("----\n");
        
        // Free C arrays
        free(matrixACArray);
        free(matrixBCArray);
        free(resultCArray);
    }
    
}

@end



int main(int argc, const char * argv[]) {
    @autoreleasepool {
        MatrixMultiplication *matrixMultiplier = [[MatrixMultiplication alloc] init];
        [matrixMultiplier multiplyMatricesWithBatchSize:16 M:1000 N:1000 K:1000];
    }
    return 0;
}