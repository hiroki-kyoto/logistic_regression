// nn.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <string.h>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <cmath>
#include <ctime>
#ifdef _WIN32_
#include "windows.h"
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "sm_20_atomic_functions.h"

using namespace std;


#define BLOCKSIZE 32


#define CUDA_CALL(func)\
  {\
    cudaError_t e = (func);\
    if(e != cudaSuccess)\
	cout << "LINE#"<<__LINE__<<": " << cudaGetErrorString(e) << endl;\
  }

struct  NN_MODEL{
	int Train_num;
	int Test_num;
	int Train_count;
	int batch_size;
	int In_nodes;
	int Hiden_nodes;
	int Out_nodes;
	float learn_r;
	float *W1;
	float *W2;
	float *B1;
	float *B2;
};




//*********************************************** GPU function ***********************************//

/**
* ���ܣ�GPU���� C = sigmod( A��B + B_D )
* ���룺dev_A A����
* ���룺dev_B B����
* �����dev_C C����
* ���룺A_height A�ĸ߶�
* ���룺A_width A�Ŀ��
* ���룺B_height B�ĸ߶�
* ���룺B_width B�Ŀ��
* ���룺B_D ƫ������
*/
__global__ void BP_Calculate_mmul(float *dev_A, float *dev_B, float *dev_C, int A_height, int A_width, int B_height, int B_width, float* B_D)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // ������
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // ������
	// ÿһ���̼߳���Csub�е�һ��Ԫ�أ����������Cvalue
	float Cvalue = 0;
	// A�����ӿ� * B�����ӿ� = ��ӦC���ӿ�Csub
	for (int m = 0; m < A_width; m += BLOCKSIZE)
	{
		int colA = m + threadIdx.x; // ��ǰ�߳��� A �е�������
		int rowB = m + threadIdx.y; // ��ǰ�߳��� B �е�������

		// ���乲���ڴ�ռ䣬�������Asub��Bsub
		__shared__ float As[BLOCKSIZE][BLOCKSIZE];
		__shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

		// ��Asub��Bsub�����������ڴ���
		if ((colA < A_width) && (y_id < A_height))
			As[threadIdx.y][threadIdx.x] = dev_A[y_id * A_width + colA]; // A(y_id, colA)
		else
			As[threadIdx.y][threadIdx.x] = 0.0f;

		if ((x_id < B_width) && (rowB < B_height))
			Bs[threadIdx.y][threadIdx.x] = dev_B[rowB * B_width + x_id]; // B(rowB, x_id)
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		// A�ӿ����*B�ӿ����
		// �ӿ��ڵ�ѭ��
		for (int idx = 0; idx < BLOCKSIZE; ++idx)
		{
			Cvalue += As[threadIdx.y][idx] * Bs[idx][threadIdx.x];
		}

		// ͬ��,ȷ����ǰA�ӿ���B�ӿ�ļ������
		__syncthreads();
	}

	if (x_id < B_width && y_id < A_height)
	{
		dev_C[y_id*B_width + x_id] = 1 / ((1 + exp(-(Cvalue + B_D[x_id]))));
		//gpu_sigmod(dev_C, Cvalue + B_D[x_id], y_id*B_width + x_id);
	}
}




/**
* ���ܣ�GPU���� �������޸��� C = A *(1-A) *(B-A)
* ���룺dev_A ���������
* ���룺dev_B ��ǩ����
* �����dev_C �������޸�������
* ���룺height �߶�
* ���룺width ���
*/
__global__ void BP_Calculate_out_update(float *dev_A, float *dev_B, float *dev_C, int height, int width)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;//������
	int col = blockDim.x * blockIdx.x + threadIdx.x; //������

	if (row < height && col < width)
	{
		dev_C[row*width + col] = dev_A[row*width + col] * (1 - dev_A[row*width + col])*(dev_B[row*width + col] - dev_A[row*width + col]);
	}
}




/**
* ���ܣ�GPU���� ���ز���޸��� D =  (A��B') * C *(1-C)
* ���룺dev_A �������޸�������
* ���룺dev_B ���ز㵽������Ȩֵ����
* ���룺dev_C ���ز�����
* �����dev_D ���ز���޸�������
* ���룺A_height A�ĸ߶�
* ���룺A_width A�Ŀ��
* ���룺B_height B�ĸ߶�
* ���룺B_width B�Ŀ��
*/
__global__ void BP_Calculate_hiden_update(float *dev_A, float *dev_B, float *dev_C, float *dev_D, int A_height, int A_width, int B_height, int B_width)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // ������
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // ������

	// ÿһ���̼߳���Csub�е�һ��Ԫ�أ����������Cvalue
	float Cvalue = 0;

	// A�����ӿ� * B�����ӿ� = ��ӦC���ӿ�Csub
	for (int m = 0; m < A_width; m += BLOCKSIZE)
	{
		int colA = m + threadIdx.x; // ��ǰ�߳��� A �е�������
		int rowB = m + threadIdx.y; // ��ǰ�߳��� B �е�������

		// ���乲���ڴ�ռ䣬�������Asub��Bsub
		__shared__ float As[BLOCKSIZE][BLOCKSIZE];
		__shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

		// ��Asub��Bsub�����������ڴ���
		if ((colA < A_width) && (y_id < A_height))
			As[threadIdx.y][threadIdx.x] = dev_A[y_id * A_width + colA]; // A(y_id, colA)
		else
			As[threadIdx.y][threadIdx.x] = 0.0f;

		if ((x_id < B_height) && (rowB <B_width))
			Bs[threadIdx.y][threadIdx.x] = dev_B[x_id * B_width + rowB]; // B(rowB, x_id)
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		// A�ӿ����*B�ӿ����
		// �ӿ��ڵ�ѭ��
		for (int idx = 0; idx < BLOCKSIZE; ++idx)
		{
			Cvalue += As[threadIdx.y][idx] * Bs[idx][threadIdx.x];
		}

		// ͬ��,ȷ����ǰA�ӿ���B�ӿ�ļ������
		__syncthreads();
	}

	if (x_id < B_height && y_id < A_height)
	{
		//ChgH[batch_size][Hiden_nodes] = temp[batch_size][Hiden_nodes] .* O1[batch_size][Hiden_nodes] .*(1-O1[batch_size][Hiden_nodes])
		dev_D[y_id * B_height + x_id] = Cvalue*dev_C[y_id * B_height + x_id] * (1 - dev_C[y_id * B_height + x_id]);
	}
}




/**
* ���ܣ�GPU����Ȩ����  C = C + lr *(A'��B);
* ���룺dev_A ��һ������
* ���룺dev_B �ò���޸���
* ���룺dev_C ��һ�㵽�ò��Ȩ����
* �����dev_C ��һ�㵽�ò��Ȩ����
* ���룺A_height A�ĸ߶�
* ���룺A_width A�Ŀ��
* ���룺B_height B�ĸ߶�
* ���룺B_width B�Ŀ��
*/
__global__ void BP_Calculate_W_update(float *dev_A, float *dev_B, float *dev_C, int A_height, int A_width, int B_height, int B_width, float learn_r)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // ������
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // ������

	// ÿһ���̼߳���Csub�е�һ��Ԫ�أ����������Cvalue
	float Cvalue = 0;

	// A�����ӿ� * B�����ӿ� = ��ӦC���ӿ�Csub
	for (int m = 0; m < A_height; m += BLOCKSIZE)
	{
		int colA = m + threadIdx.x; // ��ǰ�߳��� A �е�������
		int rowB = m + threadIdx.y; // ��ǰ�߳��� B �е�������

		// ���乲���ڴ�ռ䣬�������Asub��Bsub
		__shared__ float As[BLOCKSIZE][BLOCKSIZE];
		__shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

		// ��Asub��Bsub�����������ڴ���
		if ((colA < A_height) && (y_id < A_width))
			As[threadIdx.y][threadIdx.x] = dev_A[colA * A_width + y_id]; // A(y_id, colA)
		else
			As[threadIdx.y][threadIdx.x] = 0.0f;
		if ((x_id < B_width) && (rowB < B_height))
			Bs[threadIdx.y][threadIdx.x] = dev_B[rowB * B_width + x_id]; // B(rowB, x_id)
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		// A�ӿ����*B�ӿ����
		// �ӿ��ڵ�ѭ��
		for (int idx = 0; idx < BLOCKSIZE; ++idx)
		{
			Cvalue += As[threadIdx.y][idx] * Bs[idx][threadIdx.x];
		}

		// ͬ��,ȷ����ǰA�ӿ���B�ӿ�ļ������
		__syncthreads();
	}

	if (x_id < B_width && y_id < A_width)
	{
		dev_C[y_id * B_width + x_id] += Cvalue * learn_r;
	}
}





/**
* ���ܣ�GPU����ƫ��
* ���룺dev_A �ò���޸���
* ���룺dev_B ƫ������
* ���룺A_height A�ĸ߶�
* ���룺A_width A�Ŀ��
*/
__global__ void BP_Calculate_B_update(float *A, float*B, int A_height, int A_width, float learn_r)
{
	int col = threadIdx.x; //��

	float sum = 0.0;
	for (int i = 0; i < A_height; i++)
	{
		sum += A[i*A_width+col];
	}

	if (col < A_width)
	{
		B[col] = sum * learn_r;
	}
}

//***********************************GPU function end ***************************************//



//read data
void read_data(int data_num, int in_nodes, int out_nodes, float *data_x, float *data_y, char *data_x_file, char *data_y_file)
{
	FILE *fp1, *fp2;
	if ((fp1 = fopen(data_x_file, "r")) == NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	for (int i = 0; i < data_num; i++)
	{
		for (int j = 0; j < in_nodes; j++)
		{
			fscanf(fp1, "%f", &data_x[i*in_nodes + j]);
		}
	}
	fclose(fp1);

	if ((fp2 = fopen(data_y_file, "r")) == NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	for (int i = 0; i < data_num; i++)
	{
		for (int j = 0; j < out_nodes; j++)
		{
			fscanf(fp2, "%f", &data_y[i*out_nodes + j]);
		}
	}
	fclose(fp2);

	printf("read data end.\n");

}

//init class model
void init_class_model(NN_MODEL * model)
{
	model->Train_num = 1000000;
	model->Test_num = 10000;
	model->Train_count = 5;
	model->batch_size = 100;
	model->In_nodes = 100;
	model->Hiden_nodes = 50;
	model->Out_nodes = 2;
	model->learn_r = 0.1;

	//malloc
	model->W1 = (float*)malloc(model->In_nodes * model->Hiden_nodes * sizeof(float));
	model->W2 = (float*)malloc(model->Hiden_nodes * model->Out_nodes * sizeof(float));
	model->B1 = (float*)malloc(model->Hiden_nodes * sizeof(float));
	model->B2 = (float*)malloc(model->Out_nodes * sizeof(float));


	//init
	srand((unsigned)time(NULL));
	for (int i = 0; i < model->In_nodes; i++)
	{
		for (int j = 0; j < model->Hiden_nodes; j++)
		{
			model->W1[i*model->Hiden_nodes + j] = (rand()*2.0 / RAND_MAX - 1) / 2.0;
		}
	}
	for (int i = 0; i < model->Hiden_nodes; i++)
	{
		model->B1[i] = 1.0;
	}

	for (int i = 0; i < model->Hiden_nodes; i++)
	{
		for (int j = 0; j < model->Out_nodes; j++)
		{
			model->W2[i*model->Out_nodes + j] = (rand()*2.0 / RAND_MAX - 1) / 2.0;
		}
	}
	for (int i = 0; i < model->Out_nodes; i++)
	{
		model->B2[i] = 1.0;
	}

	printf("class model init end.\n");

}

//class model trian
void class_model_train(NN_MODEL *model, float *train_x, float *train_y)
{
	printf("feature extracted model training ......\n");



	int dev_num;
	cudaGetDeviceCount(&dev_num);
	if ( dev_num < 1 ) {
		fprintf(stdout, "NO GPU AVAILABLE FOR COMUPTING!\n");
		exit(1);
	}
	fprintf( stdout, "GPU NUM: %d\n", dev_num );
	srand(time(NULL));
	CUDA_CALL(cudaSetDevice(1));


	///* �����̸߳���߳̿� */
	dim3 dimBlock2D(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid2D_batch_in_hiden((model->Hiden_nodes + BLOCKSIZE - 1) / dimBlock2D.x, (model->batch_size + BLOCKSIZE - 1) / dimBlock2D.y); //�������
	dim3 dimGrid2D_batch_hiden_out((model->Out_nodes + BLOCKSIZE - 1) / dimBlock2D.x, (model->batch_size + BLOCKSIZE - 1) / dimBlock2D.y); //�������
	dim3 dimGrid2D_in_hiden((model->Hiden_nodes + BLOCKSIZE - 1) / dimBlock2D.x, (model->In_nodes + BLOCKSIZE - 1) / dimBlock2D.y); //�������
	dim3 dimGrid2D_hiden_out((model->Out_nodes + BLOCKSIZE - 1) / dimBlock2D.x, (model->Hiden_nodes + BLOCKSIZE - 1) / dimBlock2D.y); //�������


	//**************�����豸�˿ռ�*********************
	int Train_num = model->Train_num;
	//����
	float *train_x_D, *train_y_D;
	cudaMalloc((void**)&train_x_D, Train_num*model->In_nodes * sizeof(float));
	cudaMalloc((void**)&train_y_D, Train_num*model->Out_nodes * sizeof(float));
	//Ȩֵ
	float *W1_D, *W2_D;
	cudaMalloc((void**)&W1_D, model->In_nodes*model->Hiden_nodes * sizeof(float));
	cudaMalloc((void**)&W2_D, model->Hiden_nodes *model->Out_nodes* sizeof(float));
	//ƫ��
	float *B1_D, *B2_D;
	cudaMalloc((void**)&B1_D, model->Hiden_nodes * sizeof(float));
	cudaMalloc((void**)&B2_D, model->Out_nodes * sizeof(float));

	//���ز�����������
	float * O1_D, *O2_D;
	cudaMalloc((void**)&O1_D, model->batch_size * model->Hiden_nodes * sizeof(float));
	cudaMalloc((void**)&O2_D, model->batch_size * model->Out_nodes * sizeof(float));
	//���������ز��Ȩֵ�޸���
	float * ChgO_D, *ChgH_D;
	cudaMalloc((void**)&ChgO_D, model->batch_size * model->Out_nodes * sizeof(float));
	cudaMalloc((void**)&ChgH_D, model->batch_size * model->Hiden_nodes * sizeof(float));

	printf("device memory malloc end!\n");

	//**************�������豸����*********************
	cudaMemcpy(train_x_D, train_x, Train_num*model->In_nodes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(train_y_D, train_y, Train_num*model->Out_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(W1_D, model->W1, model->In_nodes*model->Hiden_nodes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(W2_D, model->W2, model->Hiden_nodes *model->Out_nodes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B1_D, model->B1, model->Hiden_nodes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B2_D, model->B2, model->Out_nodes*sizeof(float), cudaMemcpyHostToDevice);

	printf("copy data from CPU to GPU end!\n");

	/* ��¼ʱ�� */
	cudaEvent_t start_GPU_t, end_GPU_t;
	float elaspsedTime_t;
	cudaEventCreate(&start_GPU_t);
	cudaEventCreate(&end_GPU_t);
	cudaEventRecord(start_GPU_t, 0);


	//ѵ�����
	for (int t_c = 0; t_c < model->Train_count; t_c++)
	{
		
		/* ��¼ʱ�� */
		cudaEvent_t start_GPU, end_GPU;
		float elaspsedTime;
		cudaEventCreate(&start_GPU);
		cudaEventCreate(&end_GPU);
		cudaEventRecord(start_GPU, 0);

		//ÿ��ѵ������
		for (int index = 0; index < (Train_num - model->batch_size); index += model->batch_size)
		{
			//���ز����
			//O1[batch_size][Hiden_nodes] = S (train_X[model->batch_size][In_nodes] �� W1[In_nodes][Hiden_nodes] + B1_D[Hiden_nodes])
			BP_Calculate_mmul << <dimGrid2D_batch_in_hiden, dimBlock2D >> >(&train_x_D[index*model->In_nodes], W1_D, O1_D, model->batch_size, model->In_nodes, model->In_nodes, model->Hiden_nodes, B1_D);

			//��������  
			//O2[batch_size][Out_nodes] = S (O1[batch_size][Hiden_nodes] �� W2[Hiden_nodes][Out_nodes] + B2_D[Out_nodes])
			BP_Calculate_mmul << <dimGrid2D_batch_hiden_out, dimBlock2D >> >(O1_D, W2_D, O2_D, model->batch_size, model->Hiden_nodes, model->Hiden_nodes, model->Out_nodes, B2_D);
			   

			//����������Ȩֵ�޸���
			//ChgO[batch_size][Out_nodes] = O2[batch_size][Out_nodes] * (1 - O2) * (train_Y[batch_size][Out_nodes] - O2);
			BP_Calculate_out_update << <dimGrid2D_batch_hiden_out, dimBlock2D >> >(O2_D, &train_y_D[index*model->Out_nodes], ChgO_D, model->batch_size, model->Out_nodes);


			//�������ز��Ȩ�޸���
			//ChgH[batch_size][Hiden_nodes] = (ChgO[batch_size][Out_nodes] �� W2[Hiden_nodes][Out_nodes]') * O1[batch_size][Hiden_nodes] *(1-O1)
			BP_Calculate_hiden_update << <dimGrid2D_batch_in_hiden, dimBlock2D >> >(ChgO_D, W2_D, O1_D, ChgH_D, model->batch_size, model->Out_nodes, model->Hiden_nodes, model->Out_nodes);

			//�޸������Ȩ����
			// W2[Hiden_nodes][Out_nodes] = W2[Hiden_nodes][Out_nodes] + learn_r * O1[batch_size][Hiden_nodes] * ChgO[batch_size][Out_nodes]
			//����ת��ΪW2[Hiden_nodes][Out_nodes] = learn_r * (O1[batch_size][Hiden_nodes]'�� ChgO[batch_size][Out_nodes]);
			BP_Calculate_W_update << <dimGrid2D_hiden_out, dimBlock2D >> >(O1_D, ChgO_D, W2_D, model->batch_size, model->Hiden_nodes, model->batch_size, model->Out_nodes,model->learn_r);

			//�޸����ز�Ȩ����
			//W1[In_nodes][Hiden_nodes] = W1[In_nodes][Hiden_nodes] + learn_r * train_X[batch_size][In_nodes] * ChgH[batch_size][Hiden_nodes]
			//W1[In_nodes][Hiden_nodes] = learn_r * (train_X[batch_size][In_nodes]' �� ChgH[batch_size][Hiden_nodes])
			BP_Calculate_W_update << <dimGrid2D_in_hiden, dimBlock2D >> >(&train_x_D[index*model->In_nodes], ChgH_D, W1_D, model->batch_size, model->In_nodes, model->batch_size, model->Hiden_nodes, model->learn_r);

			//�޸�������ƫ��
			//B2[Out_nodes] = B2[Out_nodes] + learn_r * ChgO[batch_size][Out_nodes]
			BP_Calculate_B_update << <1, model->Out_nodes >> >(ChgO_D, B2_D, model->batch_size, model->Out_nodes, model->learn_r);

			//�޸����ز��ƫ��
			//B1[Hiden_nodes] = B1[Hiden_nodes] + learn_r * ChgH[batch_size][Hiden_nodes]
			BP_Calculate_B_update << <1, model->Hiden_nodes >> >(ChgH_D, B1_D, model->batch_size, model->Hiden_nodes, model->learn_r);
			//ѵ��һ���ν���
		}

		/* ��ʱ���� */
		cudaEventRecord(end_GPU, 0);
		cudaEventSynchronize(end_GPU);
		cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

		
		printf("[%d/%d], time[ms] = %0.5f\n", t_c + 1, model->Train_count, elaspsedTime);
		
	}

	printf("training end!\n");


	/* ��ʱ���� */
	cudaEventRecord(end_GPU_t, 0);
	cudaEventSynchronize(end_GPU_t);
	cudaEventElapsedTime(&elaspsedTime_t, start_GPU_t, end_GPU_t);
	printf("*******************************************\n");
	printf("total training time: %0.5f (ms).\n", elaspsedTime_t);
	printf("*******************************************\n");

	
	//�ͷ�
	cudaFree(train_x_D);
	cudaFree(train_y_D);
	cudaFree(W1_D);
	cudaFree(W2_D);
	cudaFree(O1_D);
	cudaFree(O2_D);
	cudaFree(ChgO_D);
	cudaFree(ChgH_D);

	printf("device memory free end��\n");



	printf("feature extracted model training end!\n");
}

int main()
{
	//model
	NN_MODEL *model = (NN_MODEL*)malloc(sizeof(struct NN_MODEL));
	init_class_model(model);

	//train data
	float *train_x = (float *)malloc(model->Train_num * model->In_nodes * sizeof(float));
	float *train_y = (float *)malloc(model->Train_num * model->Out_nodes * sizeof(float));
	read_data(model->Train_num, model->In_nodes, model->Out_nodes, train_x, train_y, "train_x.txt", "train_y.txt");

	//class model trian
	class_model_train(model, train_x, train_y);

	getchar();
	//free data
	free(train_x);
	free(train_y);

	free(model->W1);
	free(model->W2);
	free(model->B1);
	free(model->B2);
	free(model);

	getchar();

	return 0;
}

