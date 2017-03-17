#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <cstring>
#include <cmath>
#ifdef _WIN32
#include "windows.h"
#endif

using namespace std;

#define TARGET_ACCURACY 0.99
#define FEATURE_DIM 30
#define BATCH_SIZE 256
#define BATCH_TOTAL 1024
#define ERROR_LAST_ITERATION 16
#define LEARNING_RATE 0.01
#define LINES_TO_READ 977
#define TRAIN_RECORD_NUM 777
#define TEST_RECORD_NUM 200


// MACRO & FUNCTIONS
#define F_SIGMOID( x ) ( 1.0 / ( 1.0 + expf( -x ) ) )

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
	//float err = (float)label - sigmoid(wtx);
	float err = (float)label - F_SIGMOID( wtx );
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


// get data from dataset
void prepare_sample_batch( void * data, vector< pair<int, float> > & out, int * label ) {
    float * mat = (float*) data;
    // get input vector
    int i;
    int j = rand() % TRAIN_RECORD_NUM;
    for ( i=0; i<FEATURE_DIM-1; i++ ) {
        if ( mat[j * FEATURE_DIM + i] != 0 ) {
            out.push_back( make_pair<int, float>(i, mat[j * FEATURE_DIM + i] ) );
        }
    }
    // add bias
    out.push_back( make_pair<int, float>( FEATURE_DIM - 1, 1.0 ) );
    *label = (int) mat[j * FEATURE_DIM + FEATURE_DIM - 1];
}


void cpu_lr ( void * data, float * _model ) {
	fprintf( stdout, "============== TRAIN MODEL ===============\n" );
	float learning_rate = LEARNING_RATE;
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
		//cout << "iter#" << i << " mean error: " << err << endl;

		// update error history
		err_tot -= err_hist[err_id];
		err_tot += err;
		err_hist[err_id] = err;
		err_id = (err_id+1)%ERROR_LAST_ITERATION;

		if (err_tot/ERROR_LAST_ITERATION < 1.0 - TARGET_ACCURACY) {
			std::cout << "Target accuracy achieved!\n" << std::endl;
			break;
		}
	}
	std::cout << "training accuracy: " << 1.0-err_tot/ERROR_LAST_ITERATION << std::endl;
	// print weights
	for (i = 0; i < model.size(); i++) {
		_model[i] = model[i];
		std::cout << model[i] << " ";
	}
	std::cout << std::endl;
}


// split line into vectors of words
void split_line(
        std::vector< std::string > & words,
        const char * str,
        char delim ) {
    int i, j;
    char w[WORD_SIZE];
    words.clear();
    i = j = 0;
    for( ; i<LINE_WIDTH; i++ ) {
        if ( !str[i] ) {
            w[j] = 0;
            words.push_back( std::string( w ) );
            break;
        } else if ( str[i] == delim ) {
            w[j] = 0;
            words.push_back( std::string( w ) );
            j = 0;
        } else if ( j == WORD_SIZE - 1 ) {
            std::cout << "word size exceeds maxium limit!" << std::endl;
            exit(1);
        } else {
            w[j++] = str[i];
        }
    }
}

void clear_data( void ** data ) {
	fprintf( stdout, "=============== CLEAN DATA ===============\n" );
    delete (float*)(*data);
    *data = NULL;
}


void test_model( void * data, float * model ) {
	fprintf( stdout, "============== TEST MODEL ===============\n" );
	int i, j;
	float s, t, accuracy, precision, recall, f_value;
	int true_accept = 0;
	int false_accept = 0;
	int true_refuse = 0;
	int false_refuse = 0;

	float * mat = (float*)data;
	for ( i=0; i<TEST_RECORD_NUM; i++ ) {
		s = 0.0;
		for ( j=0; j<FEATURE_DIM - 1; j++ ) {
			s += model[j] * mat[ (TRAIN_RECORD_NUM + i) * FEATURE_DIM + j ];
		}
		s += model[FEATURE_DIM - 1];
		s = F_SIGMOID( s );
		s = s > 0.5;
		t = mat[(TRAIN_RECORD_NUM + i) * FEATURE_DIM + FEATURE_DIM - 1];
		if ( t == 1 && s == 1 ) {
			true_accept ++;
		} else if ( t == 1 && s == 0 ) {
			false_refuse ++;
		} else if ( t == 0 && s == 1 ) {
			false_accept ++;
		} else if ( t == 0 && s == 0 ) {
			true_refuse ++;
		}
	}
	// print out model analysis
	accuracy = 1.0 * (true_accept + true_refuse) / TEST_RECORD_NUM;
	precision = 1.0 * true_accept/(true_accept + false_accept);
	recall = 1.0 * true_accept/(true_accept + false_refuse);
	f_value = precision * recall * 2.0 / ( precision + recall );
	fprintf( stdout, "accuracy: %.3f.\n", accuracy );
	fprintf( stdout, "precision: %.3f.\n", precision );
	fprintf( stdout, "recall: %.3f.\n", recall );
	fprintf( stdout, "F-value: %.3f.\n", f_value );
}


int main() {
#ifdef _WIN32
	LARGE_INTEGER begin;
	LARGE_INTEGER end;
	LARGE_INTEGER freq;
	QueryPerformanceFrequency( &freq );
	QueryPerformanceCounter( &begin );
#endif
    // read data
    
	float * model = new float[FEATURE_DIM];
    read_data( "_2G_FILTERED.CSV", &data );
    // training
	// training
    // before training, set random seeds
    srand( (unsigned long) 127 );
	cpu_lr( data, model );
#ifdef _WIN32
	QueryPerformanceCounter(&end);
	double millsec = 1000.0 * (end.QuadPart - begin.QuadPart)/freq.QuadPart;
	std::cout<<"cost time: "<< millsec << " milliseconds." << std::endl;
	fflush(stdout);
#endif
    // test model
	test_model( data, model );
    // clear
    clear_data( &data );
	delete [] model;
    return 0;
}
