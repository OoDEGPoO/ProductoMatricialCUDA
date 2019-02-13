/* Enunciado:
 * Multiplicacion de Matrices MxN (16x16) por Bloques en CUDA
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

cudaError_t prodMatricesCuda(int *c, const int *a, const int *b, unsigned int Width);
const int TILE_WIDTH = 4;//Se ha establecido un tamaño de tesela de 4 hilos

__global__ void productoKernel(int *c, const int *a, const int *b, unsigned int Width)
{
	int id_fil = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int id_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int n = 0;

	if ((id_fil < Width) && (id_col < Width)) {//Si el hilo está fuera de los valores, no debe actuar
		for (int i = 0; i < Width; i++) {
			n = n + (a[id_fil*Width + i] * b[i*Width + id_col]);
		}
		c[id_fil*Width + id_col] = n;
	}
}

//Hace uso de ceil para obtener el tamaño del Grid de Bloques
int grid_calc(int Width, int Tile_Width) {
	double x =  Width;
	double y = Tile_Width;

	return (int)(ceil(x / y));//redondea hacia arriba la división
}

void imprimeMatriz(int *v, int m, int n) {//( m * n )
	int i, j, x;
	int ws;//numero de espacios de caracteres por casilla
	printf("\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ws = 5;
			x = v[i*m + j];

			if (x < 0) {//si es negativo, se ocupa un hueco por el signo "-"
				ws--;
				x = -1 * x;
			}
			else {//para alinear los dígitos
				ws--;
				printf(" ");
			}
			do {//Se ocupa un hueco por digito del numero
				ws--;
				x = x / 10;
			} while (x > 0);

			printf("%d", v[i*m + j]);//imprimimos el numero
			while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
				printf(" ");
				ws--;
			}
		}
		printf("\n");
	}
}

void imprimeMatriz(int *v, int m) {//Para matrices cuadradas ( m * m )
	int i, j, x;
	int ws;//numero de espacios de caracteres por casilla
	printf("\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			ws = 5;
			x = v[i*m + j];

			if (x < 0) {//si es negativo, se ocupa un hueco por el signo "-"
				ws--;
				x = -1 * x;
			}
			else {//para alinear los dígitos
				ws--;
				printf(" ");
			}
			do {//Se ocupa un hueco por digito del numero
				ws--;
				x = x / 10;
			} while (x > 0);

			printf("%d", v[i*m + j]);//imprimimos el numero
			while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
				printf(" ");
				ws--;
			}
		}
		printf("\n");
	}
}

void generaMatriz(int *v, int m, int n, int max, int min) {//( m * n )
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			v[i*n + j] = (rand() % (max - min)) + min;
		}
	}
}

void generaMatriz(int *v, int m, int max, int min) {//Para matrices cuadradas ( m * m )
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			v[i*m + j] = (rand() % (max - min)) + min;
		}
	}
}

int main()
{
	/*const int Width = 6;//Pruebas
	int a[6 * 6] = {	8, 7, 3, -4, -2, -3,
						8, 9, -6, 3, -1, -4,
						-6, -1, -10, -7, 8, 6,
						4, -6, -6, -3, 8, 7,
						-1, -1, -7, -8, -1, 9,
						7, 8, 3, 7, 2, 3 };

	int b[6 * 6] = {	0, 9, 9, 5, -10, 6,
						-2, 3, 8, 4, 0, -9,
						7, 1, -8, -9, -10, -9,
						-3, -5, 7, -2, 6, 4,
						7, 9, -3, -9, -9, 6,
						6, 6, 4, -8, 8, -5 };

	//Resultado de a * b =
	//		-13		80		70		91		-140	-55
	//		-100	45		200		165		-25		47
	//		45		76		-31		-50		94		53
	//		77		141		19		-72		-14		133
	//		24		66		22		7		113		-17
	//		16		91		158		-16		-52		-32*/

	srand((unsigned int)time(0));
	const int max = 10;
	const int min = -10;

    const int Width = 16;
	int a[Width * Width] = { 0 };
	generaMatriz(a, Width, max, min);

	int b[Width * Width] = { 0 };
	generaMatriz(b, Width, max, min);

    int c[Width * Width] = { 0 };
	
    // Add vectors in parallel.
    cudaError_t cudaStatus = prodMatricesCuda(c, a, b, Width);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printf("\n\tMatriz A\n");
	imprimeMatriz(a, Width);
	printf("\n\tMatriz B\n");
	imprimeMatriz(b, Width);
	printf("\n\tResultado del producto:\n");
	imprimeMatriz(c, Width);
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t prodMatricesCuda(int *c, const int *a, const int *b, unsigned int Width)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
	dim3 DimGrid(grid_calc(Width, TILE_WIDTH), grid_calc(Width, TILE_WIDTH));
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, Width * Width * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, Width * Width * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, Width * Width * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, Width * Width * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, Width * Width * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    productoKernel<<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b, Width);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, Width * Width * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
