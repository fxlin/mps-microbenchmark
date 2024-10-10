clang \
-fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders mpsmm.m    \
-DUSE_FLOAT16   \
-o mpsmm-fp16 


clang \
-fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders mpsmm.m    \
-o mpsmm-fp32