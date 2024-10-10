CC = clang
CFLAGS = -framework Foundation -framework Metal -framework MetalKit

METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

SRC = MatrixMultiplier.m


all: matrix_multiplication


matrix_multiplication.air: matrix_multiplication.metal
	$(METAL) -c $< -o $@

matrix_multiplication.metallib: matrix_multiplication.air
	$(METALLIB) $< -o $@

matrix_multiplication: MatrixMultiplier.m matrix_multiplication.metallib
	$(CC) $(CFLAGS) -o matrix_multiplication MatrixMultiplier.m

clean:
	rm -f matrix_multiplication matrix_multiplication.air matrix_multiplication.metallib