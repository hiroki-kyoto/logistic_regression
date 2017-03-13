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
    
}


#endif // READ_DATA_H
