// read_data.h
#ifndef READ_DATA_H
#define READ_DATA_H 0X01

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "def.h"

/*---------------DECLARATION ----------------*/
/***
* data : matrix to return
* info : csv table information file
* file : csv file path
* return FALSE means failure, TRUE means success
***/
int read_data_from_csv_file(
    matrix * data,
    const char * info,
    const char * file,
    const param * config
);


/***
* data : matrix to return
* info : database table information file
* host : host address(ip)
* port : oracle server port
* config : parameters used to read the table
* return : FALSE on failure, TRUE on success
***/
int read_data_from_oracle_database(
    matrix * data,
    const char * info,
    const char * host,
    int port,
    const param * config
);




/*---------------DEFINTION BEGIN ----------------*/
VALUE_TYPE check_type(char * str, int n ) {
    int i, k;
    const char * num = "NUMBER";
    str[n] = 0;
    if ( !str ) {
        exit( 1 );
    }
    for ( i=0,k=0; i<n && num[k]; ++i,++k ) {
        if ( str[i] != num[k] ) {
            return VALUE_TYPE_STRING;
        }
    }
    if ( i<n || num[k] ) {
        return VALUE_TYPE_STRING;
    }
    return VALUE_TYPE_NUMBER;
}

int find_str_in_arr( char * str, char * arr ) {
    int i, k;
    BOOL f;
    if ( str[0]==0 ) {
        exit( 1 );
    }
    for ( i=0; i<MAX_CATEGORY_NUM; ++i ) {
        if ( arr[i*VARIABLE_VALUE_LENGTH] == 0 ) {
            // fill it into the array
            memcpy( arr+(i*VARIABLE_VALUE_LENGTH), str, VARIABLE_VALUE_LENGTH * sizeof(char) );
            break;
        }
        k = 0;
        f = TRUE;
        while ( str[k] ) {
            if ( arr[i*VARIABLE_VALUE_LENGTH+k] != str[k] ) {
                f = FALSE;
                break;
            }
            ++k;
        }
        if ( f && arr[i*VARIABLE_VALUE_LENGTH+k] == 0 ) {
            break;
        }
    }
    return i;
}

// CALCULATE PEARSON COEFFICIENT
float sum(float * a, int n){
    float s = 0;
    int i;
    for ( i = 0; i < n; i++ ) {
        s += a[i];
    }
    return s;
}

/*float fabs(float x ) {
    return x>0?x:-x;
}*/

float dot_pro(float * a, float * b, int n){
    float s = 0.0;
    int i = 0;
    for (i = 0; i < n; i++){
        s += a[i] * b[i];
    }
    return s;
}

void pearson(float * T, int ncol, int nrow, float * Pears){
	// rows: attrs cols: samples
    int i, j;
	for (i = 0; i < ncol; i++){
		for (j = i + 1; j < ncol; j++) {
			Pears[i*ncol+j] =
            fabs(
                 (
                  dot_pro(T+i*nrow, T+j*nrow, nrow) -
                  sum(T+i*nrow, nrow) * sum(T+j*nrow, nrow) / nrow
                  ) /
                 sqrt(
                      (
                       dot_pro(T+i*nrow, T+i*nrow, nrow) -
                       pow(sum(T+i*nrow, nrow), 2.0) / nrow
                       ) *
                      (
                       dot_pro(T+j*nrow, T+j*nrow, nrow) -
                       pow(sum(T+j*nrow, nrow), 2.0) / nrow
                       )
                      )
                 );
		}
	}
}

BOOL is_all(int n, node * sel, node * del){
    int i;
    i = 0;
    while ( sel->next != NULL ) {
        sel = sel->next;
        ++i;
    }
    if ( sel->data != NULL ) {
        ++i;
    }
    while ( del->next != NULL ) {
        del = del->next;
        ++i;
    }
    if ( del->data != NULL ) {
        ++i;
    }
    if (i == n-1 ) {
        return TRUE;
    } else {
        return FALSE;
    }
}

BOOL is_in(int idx, node * a){
    while ( a->next ) {
        if ( *(int*)(a->data) == idx ) {
            return TRUE;
        }
        a = a->next;
    }

    if ( a->data != NULL ) {
        if ( *(int*)(a->data) == idx ) {
            return TRUE;
        }
    }

    return FALSE;
}

void push_back(node * list, void * val, int size ) {
    while ( list->next != NULL ) { // go to the end of list
        list = list->next;
    }
    if ( list->data != NULL ) { // if the list is not empty
        list->next = (node*)malloc(sizeof(node));
        list->next->prev = list;
        list = list->next;
        list->next = NULL;
    }
    list->data = (char*)malloc(sizeof(char)*size);
    memcpy( list->data, (char*)val, sizeof(char)*size );
}

node * select(float * pears, int n) {
    float * tmp, m;
    int i, idx;
    node * sel, * del, * pnode;

    tmp = (float*)malloc((n-1)*sizeof(float));
    for (i=0; i <n-1; i++) { // pearson coefficients for x(i) -> y
        tmp[i] = pears[i*n+n-1];
    }

    idx = 0;
    m = -2;

    // create sel and del list
    sel = (node*)malloc(sizeof(node));
    del = (node*)malloc(sizeof(node));

    sel->prev = NULL;
    sel->next = NULL;
    sel->data = NULL;

    del->prev = NULL;
    del->next = NULL;
    del->data = NULL;

    while (!is_all(n, sel, del)) {
        m = -2;
        // find the biggest pearson coefficient for x(i)->y;
        for (i=0; i<n-1; i++) {
            if (!is_in(i, sel) &&
                !is_in(i, del) &&
                m < tmp[i]
            ) {
                m = tmp[i];
                idx = i;
            }
        }
        push_back( sel, &idx, sizeof(idx) ); // add the most related x(i) to model;
        for (i=0; i<n-1; i++){
            if ( i != idx ) {
                if (!is_in(i, sel) &&
                    !is_in(i, del) &&
                    !is_all(n, sel, del) &&
                    pears[i*n+idx]>m+PEARSON_THRESHOLD
                ) {
                    push_back( del, &i, sizeof(i) ); // remove those co-lineared variables
                }
            }
        }
    }
    // select the variable [y]
    i = n - 1;
    push_back( sel, &i, sizeof(i) );
    // clear memory usage
    free(tmp);
    while ( del->next != NULL ) {
        pnode = del->next;
        free( del->data );
        free( del );
        del = pnode;
    }
    if ( del->data ) {
        free( del->data );
    }
    free( del );

    return sel;
}

int read_data_from_csv_file(
    matrix * data,
    const char * info,
    const char * file,
    const param * config
) {
    // read info file and get types of variables
    FILE * fp;
    char buf[FILE_READ_BUFFER_SIZE], * str, * labels, tmp, * row_labels;
    node * pname, * ptype, * types, * names, * ptmp;
    int i, k, m, n, s, ncol, nrow, *hist;
    double miu, delta, p;
    float * rows, * pears;

    pname = (node*)malloc( sizeof(node) );
    pname->prev = 0;
    pname->data = 0;
    pname->next = 0;

    ptype = (node*)malloc( sizeof(node) );
    ptype->prev = 0;
    ptype->data = 0;
    ptype->next = 0;

    data->names = pname;
    types = ptype;

    fp = fopen( info, "rt" );

    i = 0;
    while ( !feof( fp ) ) {
        buf[i] = fgetc( fp );
        if ( buf[i] == ' ' || buf[i] == '\t' ) {
            if ( i ) { // an item ended
                if ( pname->data ) { // if it is not the first node
                    pname->next = (node*)malloc( sizeof(node) );
                    pname->next->prev = pname;
                    pname = pname->next;
                    pname->next = 0;
                }
                pname->data = (char*)malloc( sizeof(char) * (VARIABLE_NAME_LENGTH+1) );
                memcpy( pname->data, buf, i * sizeof(char) );
                ((char*)(pname->data))[i] = 0;
                i = 0;
            }
        } else if ( buf[i] == ',' || buf[i] == '\n' ) {
            if ( i ) { // a type ended
                if ( ptype->data ) { // if it is not the first node
                    ptype->next = (node*)malloc( sizeof(node) );
                    ptype->next->prev = ptype;
                    ptype = ptype->next;
                    ptype->next = 0;
                }
                ptype->data = (VALUE_TYPE*)malloc( sizeof(VALUE_TYPE) );
                *((VALUE_TYPE*)(ptype->data)) = check_type( buf, i );
                i = 0;
            }
        } else {
            if ( buf[i] != '\r' ) { // fix up for windows "\r\n".
                ++i; // string length increased
                if ( i >= VARIABLE_NAME_LENGTH ) {
                    return FALSE;
                }
            }
        }
    }
/*********************** DEBUG ONLY BEGIN **********************/
#ifdef __DEBUG__
    // check variable names read so far!
    pname = data->names;
    ptype = types;
    i = 0;
    while ( pname->next ) {
        if ( pname->data ) {
            fprintf( stdout, "%s %d\n", (char*)(pname->data), *((VALUE_TYPE*)(ptype->data)) );
            ++i;
        }
        pname = pname->next;
        ptype = ptype->next;
    }
    if ( pname->data ) {
        fprintf( stdout, "%s %d\n", (char*)(pname->data), *((VALUE_TYPE*)(ptype->data)) );
        ++i;
    }
    fprintf( stdout, "variables found: %d\n", i );
#endif
/*********************** DEBUG ONLY END **********************/
    // prepare memory
    pname = data->names;
    ptype = types;
    i = 0;
    k = 0;
    while ( pname->next ) {
        ++i;
        if ( *((VALUE_TYPE*)(ptype->data)) == VALUE_TYPE_STRING ) {
            ++k;
        }
        pname = pname->next;
        ptype = ptype->next;
    }
    if ( pname->data ) {
        ++i;
        if ( *((VALUE_TYPE*)(ptype->data)) == VALUE_TYPE_STRING ) {
            ++k;
        }
    }
    str = (char*)malloc( sizeof(char) * VARIABLE_VALUE_LENGTH * MAX_CATEGORY_NUM * k );
    memset( str, 0, VARIABLE_VALUE_LENGTH * MAX_CATEGORY_NUM * k );
    if ( !str ) {
        exit( 1 );
    }
    // prepare matrix
    data->col_num = i;
    labels = (char*)malloc(sizeof(char)*data->col_num);
    for ( i=0; i<data->col_num; ++i ) {
        labels[i] = 1;
    }
    // check how many lines to read
    fclose( fp );
    fp = fopen( file, "rt" );
    if ( !fp ) {
        return FALSE;
    }
    i = 0;
    while ( !feof( fp ) ) {
        if ( fgetc( fp ) == CSV_ROW_SEG ) {
            ++i;
        }
    }
    data->row_num = i-1; // the head line contains columns of variable names
    data->data = (float*)malloc( sizeof(float)*data->col_num*data->row_num );
    row_labels = (char*)malloc( sizeof(char)*data->row_num );
#ifdef __DEBUG__
    fprintf( stdout, "ROW: %d\t COL: %d\n", data->row_num, data->col_num );
#endif
    // read data and abandon some bad variables
    i = 0;
    k = 0;
    m = 0;
    ptype = types;
    // rewind the file pointer to second line
    fseek( fp, 0, SEEK_SET );
    while ( fgetc(fp) != CSV_ROW_SEG );
    while ( !feof( fp ) ) {
        buf[i] = fgetc( fp );
        if ( buf[i] == CSV_COL_SEG || buf[i] == CSV_ROW_SEG ) {
            tmp = buf[i];
            buf[i] = 0;
            if ( *((int*)(ptype->data)) == VALUE_TYPE_NUMBER ) {
                data->data[k] = atof( buf );
            } else { // type = VALUE_TYPE_STRING
                if ( labels[k%data->col_num] ) {
                    data->data[k] = find_str_in_arr(buf, str + (VARIABLE_VALUE_LENGTH * MAX_CATEGORY_NUM * m) );
                    if ( data->data[k]>=MAX_CATEGORY_NUM ) {
                        // mark this variable as not categorical variable
                        labels[k%data->col_num] = 0;
                    }
                }
                ++m;
            }
            i = 0;
            ++k;
            ptype = ptype->next;
            // deal with row index
            if ( tmp == CSV_ROW_SEG ) {
                if ( k%(data->col_num) ) {
                    fprintf( stdout, "csv data format illegal!\n" );
                    return FALSE; // format error
                }
                ptype = types;
                m = 0;
            }
        } else {
            if ( buf[i] != '\r' ) { // fix up for windows "\r\n".
                ++i; // string length increased
                if ( i >= VARIABLE_VALUE_LENGTH ) {
                    return FALSE;
                }
            }
        }
    }
    fclose( fp );
#ifdef __DEBUG__
    // check used and unsued variables
    pname = data->names;
    fprintf( stdout, "=========ABANDON VARIABLES==========\n" );
    for ( i=0; i<data->col_num; ++i,pname=pname->next ) {
        if ( !labels[i] ) {
            fprintf( stdout, "%s\n", (char*)(pname->data) );
        }
    }
    fprintf( stdout, "====================================\n" );
#endif
    // reduce bad samples using 3-delta principle
    // initialize the row labels
    for ( i=0; i<data->row_num; ++i ) {
        row_labels[i] = 1;
    }
    for ( i=0; i<data->col_num; ++i ) {
        miu = 0;
        delta = 0;
        for ( k=0; k<data->row_num; ++k ) {
            miu += data->data[ k*data->col_num + i ];
        }
        miu /= data->row_num;
        for ( k=0; k<data->row_num; ++k ) {
            p = data->data[k*data->col_num+i] - miu;
            delta += p*p;
        }
        delta /= data->row_num;
        delta = sqrt(delta);
        for ( k=0; k<data->row_num; ++k ) {
            p = data->data[k*data->col_num+i];
            if ( p < miu-DELTA_FACTOR*delta || p > miu+DELTA_FACTOR*delta ) {
                row_labels[k] = 0; // mark current row as abandoned
            }
        }
    }

    // reduce those non-variance variables
    for ( i=0; i<data->col_num; ++i ) {
        p = data->data[i];
        m = 0; // no different
        for ( k=1; k<data->row_num; ++k ) {
            if ( row_labels[k] ) {
                if ( data->data[k*data->col_num+i] != p ) {
                    m = 1; // different value found
                    break;
                }
            }
        }
        if ( !m ) { // ずっと変わらない
            labels[i] = 0; // mark this column as abandoned
        }
    }
#ifdef __DEBUG__
    // check abandoned rows
    fprintf( stdout, "=========ABANDON ROWS==========\n" );
    k = 0;
    for ( i=0; i<data->row_num; ++i ) {
        if ( !row_labels[i] ) {
            fprintf( stdout, "%d ", i );
            ++k;
            if ( k%8==7 ) {
                fprintf( stdout, "\n" );
            }
            fflush( stdout );
        }
    }
    fprintf( stdout, "\nTOTAL: [%d] ROWS.\n", k );
    fprintf( stdout, "===============================\n" );
    // check abandoned columns
    pname = data->names;
    nrow = data->row_num - k;
    k = 0;
    fprintf( stdout, "=========ABANDON COLUMNS==========\n" );
    for ( i=0; i<data->col_num; ++i,pname=pname->next ) {
        if ( !labels[i] ) {
            fprintf( stdout, "%s\n", (char*)(pname->data) );
            ++k;
        }
    }
    fprintf( stdout, "TOTAL: [%d] COLUMNS.\n", k );
    fprintf( stdout, "==================================\n" );
#endif
    // expand the string type variables
    m = 0;
    ncol = data->col_num - k;
    ptype = types;
    pname = data->names;
    hist = (int*) malloc( sizeof(int)*data->col_num );
    memset( (char*)hist, 0, sizeof(int)*data->col_num );
    fprintf( stdout, "=============EXPAND VARIABLES============\n" );
    for ( i=0; i<data->col_num; ++i ) {
        if ( labels[i] ) {
            if ( *(VALUE_TYPE*)(ptype->data) == VALUE_TYPE_NUMBER ) {
                fprintf( stdout, "%s\n", (char*)pname->data );
                hist[i] = -1;
                ++m;
            } else { // VALUE_TYPE_STRING
                n = 0; // max id
                for ( k=0; k<data->row_num; ++k ) {
                    if ( row_labels[k] ) {
                        if ( data->data[k*data->col_num+i] > n ) {
                            n = data->data[k*data->col_num+i];
                        }
                    }
                }
                hist[i] = n; // prevent from the co-linearity
                m += hist[i];
                fprintf( stdout, "%s[0-%d]\n", (char*)pname->data, n );
            }
        }
        ptype = ptype->next;
        pname = pname->next;
    }
    fprintf( stdout, "EXPAND FROM %d TO %d VARIABLES!\n", ncol, m );
    // create new matrix for expanded VARIABLES
    ncol = m;
    rows = (float*)malloc( sizeof(float)*nrow*ncol );
    s = 0;  // index of column
    for ( i=0; i<data->col_num; ++i ) {
        n = 0;  // index of row
        if ( hist[i] > 0 ) {    // STRING
            for ( k=0; k<data->row_num; ++k ) {
                if ( row_labels[k] ) {
                    for ( m=0; m<hist[i]; ++m ) {
                        if ( data->data[k*data->col_num+i]==m ) {
                            rows[(s+m)*nrow+n] = 1;
                        } else {
                            rows[(s+m)*nrow+n] = 0;
                        }
                    }
                    ++n;
                }
            }
            s += hist[i];
        } else if ( hist[i] < 0 ) { // NUMBER
            for ( k=0; k<data->row_num; ++k ) {
                if ( row_labels[k] ) {
                    rows[s * nrow + n] = data->data[k*data->col_num+i];
                    ++n;
                }
            }
            ++s;
        }
    }
    fprintf( stdout, "=========================================\n" );
//============ clear memory usage============
    // clear labels
    free( labels );
    free( row_labels );
    // clear data
    free( data->data );
//=========================================
//==========rebuild the name list==========
    names = data->names; // no wild pointer
    pname = data->names;
    data->names = (node*)malloc(sizeof(node));
    data->names->data = NULL;
    data->names->prev = NULL;
    data->names->next = NULL;
    ptmp = data->names; // iterator for types
    ptype = types;
    m = 0; // counter for string array
    for ( i=0; i<data->col_num; ++i ) {
        labels = (char*)pname->data;
        if ( hist[i] < 0 ){ // NUMBER
            if ( ptmp->data != NULL ) { // not first name
                ptmp->next = (node*)malloc(sizeof(node));
                ptmp->next->prev = ptmp;
                ptmp = ptmp->next;
                ptmp->next = NULL;
            }
            ptmp->data = (char*)malloc(sizeof(char)*VARIABLE_NAME_LENGTH);
            sprintf( ptmp->data, "%s", labels );
        } else if ( hist[i] > 0 ) {
            for( k=0; k<hist[i]; ++k ) {
                if ( ptmp->data != NULL ) { // not first name
                    ptmp->next = (node*)malloc(sizeof(node));
                    ptmp->next->prev = ptmp;
                    ptmp = ptmp->next;
                    ptmp->next = NULL;
                }
                ptmp->data = (char*)malloc(sizeof(char)*(VARIABLE_NAME_LENGTH+8+VARIABLE_VALUE_LENGTH));
                sprintf(
                    ptmp->data,
                    "%s[%s]",
                    labels,
                    str+(m*VARIABLE_VALUE_LENGTH*MAX_CATEGORY_NUM+k*VARIABLE_VALUE_LENGTH)
                );
            }
        }
        if ( *(VALUE_TYPE*)(ptype->data) == VALUE_TYPE_STRING ) {
            ++m;
        }
        ptype = ptype->next;
        pname = pname->next;
    }
    // update matrix
    data->col_num = ncol;
    data->row_num = nrow;
    data->data = rows; // this is a transformed matrix on ROWS!!!
    // clear hist
    free( hist );
    // clear str
    free( str );
    // clear types
    while ( types->next != NULL ) {
        ptype = types->next;
        free( types->data );
        free( types );
        types = ptype;
    }
    free( types->data );
    free( types );
    // clear names
    while ( names->next != NULL ) {
        pname = names->next;
        free( names->data );
        free( names );
        names = pname;
    }
    free( names->data );
    free( names );
//==========================================
#ifdef __DEBUG__
    pname = data->names;
    i = 0;
    while ( pname->next != NULL ) {
        fprintf( stdout, "[%3d] %s\n", i+1, (char*)pname->data );
        ++i;
        pname = pname->next;
    }
    fprintf( stdout, "[%3d] %s\n", i+1, (char*)pname->data );
#endif // __DEBUG__
    //=============NORMALIZATION===============
    if ( config->norm_method == NORM_MIN_MAX ) {
        for ( i=0; i<data->col_num; ++i ) {
            miu = data->data[i*data->row_num]; // min
            delta = miu; // max
            for ( k=0; k<data->row_num; ++k ) {
                if ( data->data[i*data->row_num+k] < miu ) {
                     miu = data->data[i*data->row_num+k];
                } else if (  data->data[i*data->row_num+k] > delta ) {
                    delta =  data->data[i*data->row_num+k];
                }
            }
            for ( k=0; k<data->row_num; ++k ) {
                data->data[i*data->row_num+k] = (data->data[i*data->row_num+k]-miu)/(delta-miu);
            }
        }
    }
    fprintf( stdout, "=============NORMALIZATION DONE==============\n" );
    //=========================================
    //==============VARIABLE SELECTION=================
    fprintf( stdout, "=============VARIABLE SELECTION==============\n" );
    // create pearson coefficient matrix
    pears = (float*)malloc(sizeof(float)*data->col_num*data->col_num);
    memset((char*)pears, 0, sizeof(float)*data->col_num*data->col_num);
    pearson(data->data, data->col_num, data->row_num, pears);
    ptmp = select(pears, data->col_num); // selected ids are stored.
    ptype = ptmp; // save to release this memory block
    labels = (char*)malloc(sizeof(char)*data->col_num);
    memset(labels, 0, sizeof(char)*data->col_num );
    ncol = 0;
    while ( ptmp->next != NULL ) {
        labels[*(int*)(ptmp->data)] = 1;
        ptmp = ptmp->next;
        ++ncol;
    }
    if ( ptmp->data != NULL ) {
        labels[*(int*)(ptmp->data)] = 1;
        ++ncol;
    }
    if ( ncol == 0 ) { // in case no variables enter the model
        fprintf( stdout, "NO VARIABLES SELECTED, MODEL FAILED!\n" );
        exit(1);
    }
    pname = data->names;
    ptmp = NULL;    // save the last name
    fprintf(stdout, "SELECTED COLUMNS[%d]: \n", ncol);
    for (i=0; i<data->col_num; ++i ) {
        if (labels[i]) {
            fprintf(stdout, "%s\n", (char*)(pname->data));
            if ( ptmp == NULL ) { // HEAD NODE!
                data->names = pname;
            } else {
                ptmp->next = pname;
            }
            pname->prev = ptmp;
            ptmp = pname; // update the last name
            pname = pname->next;
        } else {
            // release nodes
            types = pname->next;
            free( pname->data );
            free( pname );
            pname = types;
        }
    }
    // rebuild matrix in columns
    rows = data->data;
    data->data = (float*)malloc(sizeof(float)*ncol*data->row_num);
    m = 0; // real column index for new matrix
    for ( i=0; i<data->col_num; ++i ) {
        if ( labels[i] ) {
            for ( k=0; k<data->row_num; ++k ) {
                data->data[k*ncol+m] = rows[i*data->row_num+k];
            }
            ++m;
        }
    }
    data->col_num = ncol;
    // clear memory usage
    free( pears );
    free( rows );
    while ( ptype->next != NULL ) {
        ptmp = ptype->next;
        free(ptype->data);
        free(ptype);
        ptype = ptmp;
    }
    if ( ptype->data != NULL ) {
        free( ptype->data );
    }
    free( ptype );
    free(labels);

    fprintf( stdout, "=============================================\n" );
    //===================================================


    return TRUE;
}

#endif // READ_DATA_H
