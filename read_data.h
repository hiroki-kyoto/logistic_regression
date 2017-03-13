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
int read_data_from_csv_file( matrix * data, const char * info, const char * file ) {
    // read info file and get types of variables
    FILE * fp;
    char buf[FILE_READ_BUFFER_SIZE];
    node * name_list;
    int i, k, m, n, p;
    fp = fopen( file, "rt" );
    name_list = (node*)malloc(sizeof(node));
    name_list->prev = 0;
    name_list->data = 0;
    name_list->next = 0;
    while ( !feof( fp )) {
        n = fread( buf, FILE_READ_BUFFER_SIZE, sizeof(char), fp );
        k = 0; // flag for 'within an item'
        m = 0; // flag for 'within an variable'
        p = -1; // last occurence index of segment
        for ( i=0; i<n; i++ ) {
            if ( buf[i] == ' ' || buf[i] == '\t' ) {
                if ( k ) { // an item ended
                    name_list->data = (char*) malloc( sizeof(char) * VARIABLE_NAME_LENGTH );
                    memcpy( name_list->data, buf + ( p+ 1 ), sizeof(char) * ( i - p ) );
                    k = 0;
                }
                p = i;
            } else if ( buf[i] == ',' || buf[i] == '\n' ) {
                if ( m ) { // an variable ended
                    m = 0;
                }
                p = i;
            } else {
                k = m = 1;
            }
        }
        if ( n<FILE_READ_BUFFER_SIZE ) {
            break; // read over
        }
    }
}


#endif // READ_DATA_H
