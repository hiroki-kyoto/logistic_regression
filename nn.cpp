// nn.cpp : 定义控制台应用程序的入口点。
//

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

using namespace std;

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

//create data
void create_data(int row, int col, char * data_x_file, char * data_y_file)
{
	FILE *fp1, *fp2;
	if ((fp1 = fopen(data_x_file, "w")) == NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	if ((fp2 = fopen(data_y_file, "w")) == NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	float sum = 0.0;
	for (int i = 0; i < row; i++)
	{
		sum = 0.0;
		for (int j = 0; j < (col-1); j++)
		{
			//0-1 rand
			float tmp = rand()/ (RAND_MAX + 1.0);
			fprintf(fp1,"%0.2f ",tmp);
			sum += tmp;
		}
		float tmp = rand() / (RAND_MAX + 1.0);
		sum += tmp;
		fprintf(fp1, "%0.2f\n", tmp);

		if (sum <= (col * 1.0 / 2.0))
		{
			fprintf(fp2, "%d %d\n", 1,0);
		}
		else
		{
			fprintf(fp2, "%d %d\n", 0, 1);
		}
	}
	fclose(fp1);
	fclose(fp2);
}

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
	printf("class model training......\n");
	float mse_error = 0;
	double start = 0;
	double end = 0;
	float *average = (float*)malloc(model->batch_size * sizeof(float));

	//malloc
	float *train_X = (float*)malloc(model->batch_size * model->In_nodes * sizeof(float));
	float *train_Y = (float*)malloc(model->batch_size * model->Out_nodes * sizeof(float));

	float *O1 = (float*)malloc(model->batch_size * model->Hiden_nodes * sizeof(float));
	float *O2 = (float*)malloc(model->batch_size * model->Out_nodes * sizeof(float));
	float * Chg1 = (float *)malloc(model->batch_size * model->Hiden_nodes * sizeof(float));
	float * Chg2 = (float*)malloc(model->batch_size * model->Out_nodes * sizeof(float));

	//training
	for (int t_c = 0; t_c < model->Train_count; t_c++)
	{
		//mse error set 0
		mse_error = 0.0;
		// time start
		start = clock();

		// batch training
		for (int index = 0; index < (model->Train_num - model->batch_size); index += model->batch_size)
		{
			//copy batch data
			memset(train_X, 0, model->batch_size * model->In_nodes * sizeof(float));
			memset(train_Y, 0, model->batch_size * model->Out_nodes * sizeof(float));
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->In_nodes; j++)
				{
					train_X[i*model->In_nodes + j] = train_x[index*model->In_nodes + i*model->In_nodes + j];
				}
			}
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Out_nodes; j++)
				{
					train_Y[i*model->Out_nodes + j] = train_y[index*model->Out_nodes + i*model->Out_nodes + j];
				}
			}

			/* hidden output
			* O1[batch_size][Hiden_nodes] = s(train_X[batch_size][In_nodes] × W1[In_nodes][Hiden_nodes] + B1[Hiden_nodes])
			*/
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Hiden_nodes; j++)
				{
					float sum = 0.0;
					for (int k = 0; k < model->In_nodes; k++)
					{
						sum += train_X[i*model->In_nodes + k] * model->W1[k*model->Hiden_nodes + j];
					}
					sum += model->B1[j];
					O1[i*model->Hiden_nodes + j] = 1 / (1 + exp(-sum));
				}
			}

			//output   
			//O2[batch_size][Out_nodes] = sigmod ( O1[batch_size][Hiden_nodes] × W2[Hiden_nodes][Out_nodes] + B2[Out_nodes])
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Out_nodes; j++)
				{
					float sum = 0.0;
					for (int k = 0; k < model->Hiden_nodes; k++)
					{
						sum += O1[i*model->Hiden_nodes + k] * model->W2[k*model->Out_nodes + j];
					}
					sum += model->B2[j];
					O2[i*model->Out_nodes + j] = 1 / (1 + exp(-sum));
				}
			}

			//output change
			//Chg2[batch_size][Out_nodes] = O2[batch_size][Out_nodes] * (1 - O2) * (train_Y[batch_size][Out_nodes] - O2);
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Out_nodes; j++)
				{
					Chg2[i*model->Out_nodes + j] = O2[i*model->Out_nodes + j] * (1 - O2[i*model->Out_nodes + j]) * (train_Y[i*model->Out_nodes + j] - O2[i*model->Out_nodes + j]);
				}
			}

			//hiden change
			//Chg1[batch_size][Hiden_nodes] = ( Chg2[batch_size][Out_nodes] × W2[Hiden_nodes][Out_nodes]' ) * O1[batch_size][Hiden_nodes] *(1-O1)
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Hiden_nodes; j++)
				{
					float sum = 0.0;
					for (int k = 0; k < model->Out_nodes; k++)
					{
						sum += Chg2[i*model->Out_nodes + k] * model->W2[k*model->Hiden_nodes + j];
					}
					Chg1[i*model->Hiden_nodes + j] = sum * O1[i*model->Hiden_nodes + j] * (1 - O1[i*model->Hiden_nodes + j]);
				}
			}

			//output layer: update W B
			// W2[Hiden_nodes][Out_nodes] += learn_r * O1[batch_size][Hiden_nodes] * Chg2[batch_size][Out_nodes]
			// B2[Out_nodes] += learn_r * Chg2[batch_size][Out_nodes]
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Hiden_nodes; j++)
				{
					for (int k = 0; k < model->Out_nodes; k++)
					{
						model->W2[j*model->Out_nodes + k] += model->learn_r * O1[i*model->Hiden_nodes + j] * Chg2[i*model->Out_nodes + k];
					}
				}
			}
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Out_nodes; j++)
				{
					model->B2[j] += model->learn_r * Chg2[i*model->Out_nodes + j];
				}
			}

			/* hiden layer: update W B
			* W1[In_nodes][Hiden_nodes] += learn_r * train_X[batch_size][In_nodes] * Chg1[batch_size][Hiden_nodes]
			* B1[Hiden_nodes] += learn_r * Chg1[batch_size][Hiden_nodes]
			*/
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->In_nodes; j++)
				{
					for (int k = 0; k < model->Hiden_nodes; k++)
					{
						model->W1[j*model->Hiden_nodes + k] += model->learn_r * train_X[i*model->In_nodes + j] * Chg1[i*model->Hiden_nodes + k];
					}
				}
			}
			for (int i = 0; i < model->batch_size; i++)
			{
				for (int j = 0; j < model->Hiden_nodes; j++)
				{
					model->B1[j] += model->learn_r * Chg1[i*model->Hiden_nodes + j];
				}
			}

			//update end
		}// batch training end

		end = clock();
		printf("[%d/%d], time[ms] = %0.1lf, learn_r = %.6f\n", t_c + 1, model->Train_count, end - start, model->learn_r);

	}//training end


	//free data
	free(average);
	free(train_X);
	free(train_Y);
	free(O1);
	free(O2);
	free(Chg1);
	free(Chg2);

	printf("feature extracted model training end!\n");
}

int main()
{
	//model
	NN_MODEL *model = (NN_MODEL*)malloc(sizeof(struct NN_MODEL));
	init_class_model(model);

	//create train data
	//create_data(model->Train_num,model->In_nodes,"train_x_t.txt","train_y_t.txt");

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

