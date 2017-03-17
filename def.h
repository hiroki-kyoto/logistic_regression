#ifndef DEF_H
#define DEF_H

typedef int BOOL;

#define TRUE	1
#define FALSE	0

#define CSV_COL_SEG					','
#define CSV_ROW_SEG					'\n'

#define COLUMN_MAX_WIDTH            64
#define VARIABLE_NAME_LENGTH        64
#define VARIABLE_VALUE_LENGTH       64
#define MAX_CATEGORY_NUM 			8
#define DELTA_FACTOR				6

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

#define FILE_READ_BUFFER_SIZE		256
#define FILE_WRITE_BUFFER_SIZE		256

// data structure of orginal data
typedef struct T_ORIGINAL_DATA {
	int col_num;
	int row_num;

	VARIABLE_TYPE * var_types;

    char ** variables;
	char *** values;
} table;

typedef table odata;

typedef struct T_NODE {
	void * data;
	struct T_NODE * prev;
	struct T_NODE * next;
} node;

typedef struct T_MATRIX {
	node * names;
    int col_num;
    int row_num;
    float * data;
} matrix;

typedef matrix mdata;

typedef struct T_SPARSE_MATRIX {
    int col_num;
    int row_num;
    int total_nonzero;
    int * col_ids;
    int * row_ids;
    float * data;
} sparse_matrix;


typedef struct T_PARAM {

} param;

#endif // DEF_H
