// FL oct 2024


/// clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders -o mpsmm mpsmm.m

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// even slower than float32!
#if defined(USE_FLOAT16)
#define DATATYPE __fp16
#define MPSDataType MPSDataTypeFloat16
#else
#define DATATYPE float
#define MPSDataType MPSDataTypeFloat32
#endif

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 4) {
            NSLog(@"Usage: program M K N");
            return -1;
        }
        
        // Parse matrix dimensions from command line arguments
        NSUInteger rowsA = (NSUInteger)atoi(argv[1]);
        NSUInteger colsA = (NSUInteger)atoi(argv[2]);
        NSUInteger rowsB = colsA;
        NSUInteger colsB = (NSUInteger)atoi(argv[3]);
        NSUInteger rowsC = rowsA;
        NSUInteger colsC = colsB;
        
        // List all devices and select a suitable one
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        id<MTLDevice> device = nil;
        id<MTLDevice> selectedDevice = nil;
        for (id<MTLDevice> dev in devices) {
            // NSLog(@"Metal device: %@", dev.name);
            if ([dev supportsFamily:MTLGPUFamilyApple1]) {
                selectedDevice = dev;
                break;
            }
        }
        if (!selectedDevice) {
            NSLog(@"No suitable Metal device found");
            return -1;
        }
        device = selectedDevice;
        // NSLog(@"Selected Metal device: %@", selectedDevice.name);
        

        // Check if the device has hardware support for float16
        if ([device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v4]) {
            // NSLog(@"The device %@ has hardware support for float16.", device.name);
            ;
        } else {
            NSLog(@"The device %@ does not have native hardware support for float16.", device.name);
        }

        // Command queue to manage the commands
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        // Create MPSMatrixDescriptors for matrices A, B, and C
        MPSMatrixDescriptor *matrixADescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsA columns:colsA rowBytes:colsA * sizeof(DATATYPE) dataType:MPSDataType];
        MPSMatrixDescriptor *matrixBDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsB columns:colsB rowBytes:colsB * sizeof(DATATYPE) dataType:MPSDataType];
        MPSMatrixDescriptor *matrixCDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsC columns:colsC rowBytes:colsC * sizeof(DATATYPE) dataType:MPSDataType];
        
        // Create GPU buffers to store matrix data
        id<MTLBuffer> bufferA = [device newBufferWithLength:rowsA * colsA * sizeof(DATATYPE) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithLength:rowsB * colsB * sizeof(DATATYPE) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:rowsC * colsC * sizeof(DATATYPE) options:MTLResourceStorageModeShared];
        
        // Fill matrices A and B with some values (for example purposes, all ones)
        DATATYPE *aPointer = (DATATYPE *)bufferA.contents;
        DATATYPE *bPointer = (DATATYPE *)bufferB.contents;
        for (NSUInteger i = 0; i < rowsA * colsA; i++) {
            aPointer[i] = 1.0;
        }
        for (NSUInteger i = 0; i < rowsB * colsB; i++) {
            bPointer[i] = 1.0;
        }
        
        // Create MPSMatrix objects for A, B, and C
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:matrixADescriptor];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:matrixBDescriptor];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:matrixCDescriptor];
        
        // Create the MPSMatrixMultiplication kernel
        MPSMatrixMultiplication *matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                                        transposeLeft:false
                                                                                       transposeRight:false
                                                                                              resultRows:rowsC
                                                                                           resultColumns:colsC
                                                                                         interiorColumns:colsA
                                                                                                  alpha:1.0
                                                                                                   beta:0.0];
        // Create a command buffer to encode the operations
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Encode the matrix multiplication operation
        [matrixMultiplication encodeToCommandBuffer:commandBuffer
                                          leftMatrix:matrixA
                                         rightMatrix:matrixB
                                         resultMatrix:matrixC];
        
        // Measure the execution time
        NSDate *startTime = [NSDate date];
        
        // Commit the command buffer and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Calculate the elapsed time
        NSDate *endTime = [NSDate date];
        NSTimeInterval executionTime = [endTime timeIntervalSinceDate:startTime] * 1000.0; // Convert to milliseconds
        NSLog(@"Execution time: %.3f ms", executionTime);
    }
    return 0;
}