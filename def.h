#ifndef DEF_H
#define DEF_H

typedef int BOOL;

#define TRUE	1
#define FALSE	0

#define COLUMN_MAX_WIDTH            64
#define VARIABLE_NAME_LENGTH        64
#define VARIABLE_VALUE_LENGTH       64

typedef int VALUE_TYPE;

#define VALUE_TYPE_NUMBER           0
#define VALUE_TYPE_STRING           1

typedef int VARIABLE_TYPE;

#define VARIABLE_TYPE_FLAG          0
#define VARIABLE_TYPE_CATEGORIAL    1
#define VARIABLE_TYPE_CONTINUOUS    2
#define VARIABLE_TYPE_NONE          3

typedef int VARIABLE_ROLE;
#define VARIABLE_ROLE_INPUT         0
#define VARIABLE_ROLE_NONE          1
#define VARIABLE_ROLE_TARGET        2

// data structure of orginal data 
typedef struct T_ORIGINAL_DATA {
	int col_num;
	int row_num;
    
	VARIABLE_TYPE * var_types;
	
    char ** variables;
	char *** values;
} odata;


typedef struct T_MATRIX {
    int col_num;
    int row_num;
    float * data;
} mdata;

typedef struct T_SPARSE_MATRIX {
    int col_num;
    int row_num;
    int total_nonzero;
    int * col_ids;
    int * row_ids;
    float * data;
} sdata;

#endif // DEF_H
