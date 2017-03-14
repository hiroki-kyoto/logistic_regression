// read_data.h
#ifndef READ_DATA_H
#define READ_DATA_H 0X01

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "def.h"

/*---------------DECLARATION ----------------*/
/***
* data : matrix to return
* info : csv table information file
* file : csv file path
* return FALSE means failure, TRUE means success
***/
int read_data_from_csv_file( matrix * data, const char * info, const char * file );


/***
* data : matrix to return
* info : database table information file
* host : host address(ip)
* port : oracle server port
* return : FALSE on failure, TRUE on success
***/
int read_data_from_oracle_database( matrix * data, const char * info, const char * host, int port );




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

int read_data_from_csv_file( matrix * data, const char * info, const char * file ) {
    // read info file and get types of variables
    FILE * fp;
    char buf[FILE_READ_BUFFER_SIZE], * str;
    node * pname, * ptype, * types;
    int i, k;

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
            ++i; // string length increased
            if ( i >= VARIABLE_NAME_LENGTH ) {
                return FALSE;
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
    if ( !str ) {
        exit( 1 );
    }
    // prepare matrix
    data->col_num = i;
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
#ifdef __DEBUG__
    fprintf( stdout, "ROW: %d\t COL: %d\n", data->row_num, data->col_num );
#endif
    // read data and abandon some bad variables
    i = 0;
    k = 0;
    ptype = types;
    // rewind the file pointer to second line
    fseek( fp, 0, SEEK_SET );
    while ( fgetc(fp) != CSV_ROW_SEG );
    while ( !feof( fp ) ) {
        buf[i] = fgetc( fp );
        if ( buf[i] == CSV_COL_SEG ) {
            buf[i] = 0;
            if ( *((int*)(ptype->data)) == VALUE_TYPE_NUMBER ) {
                data->data[] atof( buf );
            } else { // it's a string

            }
        }  else if ( buf[i] == CSV_ROW_SEG ) {
        } else {
            ++i;
            if ( i >= VARIABLE_VALUE_LENGTH ) {
                return FALSE;
            }
        }
    }

    return TRUE;
}


#endif // READ_DATA_H
