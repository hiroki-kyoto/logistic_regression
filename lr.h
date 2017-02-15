#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <cmath>
#include "windows.h"
#include <fstream>

using namespace std;

#define TARGET_ACCURACY 0.99
//#define FEATURE_DIM 100
#define FEATURE_DIM 30
//#define BATCH_SIZE 16384
#define BATCH_SIZE 1024
//#define BATCH_SIZE 4096
#define BATCH_TOTAL 100
#define ERROR_LAST_ITERATION 5

float cpu_dot(const vector< pair<int, float> > & a, const vector<float> & b) {
	float ret = 0.0;
	for (vector< pair<int, float> >::const_iterator i = a.begin();
		i != a.end(); i++) {
		ret += i->second * b[i->first];
	}
	return ret;
}

vector<float> cpu_batch_dot(const vector< vector< pair<int, float> > > & data, const vector<float> & b) {
	vector<float> rets(data.size(), 0);
	for (int i = 0; i < data.size(); i++) {
		rets[i] = cpu_dot(data[i], b);
	}
	return rets;
}

double sigmoid(float x) {
	return 1.0 / (1.0 + exp(-1.0 * x));
}

double cpu_grad(const vector< pair<int, float> > & x,
	const float wtx,
	const int label,
	vector<float> & w,
	const float learning_rate,
	const float lambda) {
	float err = (float)label - sigmoid(wtx);
	for (vector< pair<int, float> >::const_iterator i = x.begin();
		i != x.end(); i++) {
		w[i->first] += learning_rate * (err - lambda * w[i->first]);
	}
	return abs(err);
}

double cpu_batch_grad(const vector< vector< pair<int, float> > > & data,
	const vector< int > & label,
	vector<float> & b,
	const float learning_rate,
	const float lambda) {
	vector<float> dot = cpu_batch_dot(data, b);
	float err = 0.;
	float total = 0.;
	for (int i = 0; i < data.size(); i++) {
		err += cpu_grad(data[i], dot[i], label[i], b, learning_rate, lambda);
		total += 1.;
	}
	return err / total;
}


void mock_sample(const int max_feature_id, vector< pair<int, float> > & out, int * label) {
	int base = (int)floor(0.2*max_feature_id);
  int count = rand() % (1-base) + base;
  int ret = 0;
  for(int i = 0; i < count; i++) {
    int fid = rand() % max_feature_id;
    if(fid % 2 == 0) ret += 1;
    else ret -= 1;
    if(abs(ret) > 10) break;
    out.push_back(make_pair<int, float>(fid, 1.0));
  }
  *label = (ret > 0) ? 1 : 0;
}


// get data from dataset
void prepare_sample_batch( void * data, vector< pair<int, float> > & out, int * label ) {
    
}


void cpu_lr ( void * data ) {
	float learning_rate = 0.01;
	float lambda = 0.00;
	float err;
	float err_hist[ERROR_LAST_ITERATION]; // error last for a low level
	int err_id = 0;
	float err_tot = 0;
	for(int i=0; i<ERROR_LAST_ITERATION; i++)
	{
		err_hist[i] = 1.0;
		err_tot += err_hist[i];
	}
	vector<float> model(FEATURE_DIM, 0);
	// initialize model
	for (int i = 0; i < model.size(); i++) {
		model[i] = 0.5 - (double)(rand() % 10000) / 10000.0;
		//model[i] = 1.0;
	}
	// mini-batch algorithm with steepest-descent method
	size_t i, j;
	int l;
	for (i = 0; i < BATCH_TOTAL; i++) {
		vector< vector< pair<int, float> > > batch_data;
		vector< int > batch_label;
		for (j = 0; j < BATCH_SIZE; j++) {
			vector< pair<int, float> > x;
            prepare_sample_batch( data, x, &l );
			batch_data.push_back(x);
			batch_label.push_back(l);
		}
		// caculate the error
		err = cpu_batch_grad(batch_data, batch_label, model,
			learning_rate, lambda);
		cout << "iter#" << i << " mean error: " << err << endl;

		// update error history
		err_tot -= err_hist[err_id];
		err_tot += err;
		err_hist[err_id] = err;
		err_id = (err_id+1)%ERROR_LAST_ITERATION;

		if (err_tot/ERROR_LAST_ITERATION < 1.0 - TARGET_ACCURACY)
		{
			std::cout << "Target accuracy achieved!\n" << std::endl;
			break;
		}
	}
	std::cout << "final model accuracy is: " << 1.0-err_tot/ERROR_LAST_ITERATION << std::endl;
	// print weights
	for (i = 0; i < model.size(); i++)
		std::cout << model[i] << " ";
	std::cout << std::endl;
}


// read data from csv file and convert it into spared data format
void read_data( const char ** file, void ** data ) {
    // last stop date
    std::fstream strm( file );
    std::string str;
    strm.getline( str, -1 );
    std::cout<< str << std::endl;
}


int main() {
	LARGE_INTEGER begin;
	LARGE_INTEGER end;
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&begin);
    // read data
    void * data;
    read_data( "_2G_MERGED.CSV", & data );
    
    
    /*
    // training
	cpu_lr( data );
	QueryPerformanceCounter(&end);
	double millsec = 1000.0 * (end.QuadPart - begin.QuadPart)/freq.QuadPart;
	std::cout<<"cost time: "<< millsec << " milliseconds." << std::endl;
	fflush(stdout);
	// pause
	//fflush(stdin);
	//getchar();
    
    */
    
    return 0;
}
