#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "def.h"

/* function declaration */

/* data : original data pointer, the only object to return;
 * file_path : file path of input data, absolute or relative;
 * col_delim : column delimiter character, NULL means default value :  ',';
 * row_delim : row delimiter character, NULL means default value : '\n';
 * col_num : set number of columns to read, non-positive number like 0 and -1 means default value : $all;
 * row_num : set number of rows to write, non-positive number like 0 and -1 means default value : $all;
 * value_types : set value types for variables;
 * variable_types : set types for variable [CONTINUOUS, FLAG, CATEGORIAL];
 * variable_roles : set roles for variable [INPUT, TARGET, NONE]
*/
BOOL read( 
		odata * data,
		const char * file_path, 
		char col_delim, 
		char row_delim, 
		int col_num, 
		int row_num, 
		VALUE_TYPE * value_types,
        VARIABLE_TYPE * variable_types, 
        VARIABLE_ROLE * variable_roles );



/* function definition */
BOOL read( 
		odata * data,
		const char * file_path,
		char col_delim,
		char row_delim,
		int col_num,
		int row_num,
        VALUE_TYPE * value_types,
        VARIABLE_TYPE * variable_types, 
        VARIABLE_ROLE * variable_roles ) {
	
	FILE * fp = fopen( file_path, "rt" );
	
	if ( !fp ) {
		fprintf( stdout, "cannot open file [%s] ! \n", file_path );
		fflush( stdout );
		return FALSE;
	}
	
	if ( !col_delim )
		col_delim = ',';
	if ( !row_delim )
		row_delim = '\n';
	
	// read file into data structure
    char c;
	int i = 0;
    int j = 0;
    int irow = 0;
	int icol = 0;
	char str[COLUMN_MAX_WIDTH];
	int istr = 0;
	char * var_name = NULL;
	char * var_value = NULL;
    BOOL skip_to_next_row;
    VALUE_TYPE t_val;
    VARIABLE_TYPE t_var;
    VARIABLE_ROLE t_role;
    
    data->col_num = 0;
    data->row_num = 0;
    data->values = NULL;
    data->variables = NULL;
    data->var_types = NULL;
	
    // first scan to get column number
    istr = 0;
    while ( TRUE ) {
        if ( feof( fp ) ) {
            fprintf( stdout, "ERROR: file content illegal!\n" );
            fflush( stdout );
            fclose( fp );
            return FALSE;
        }
        c = fgetc( fp );
        if ( c == row_delim ) {
            data->col_num ++; // last column
            fprintf( stdout, "INFO: found [%d] columns.\n", data->col_num );
            fflush( stdout );
            break;
        }
        if ( c != col_delim ) {
            istr ++;
            if ( istr >= COLUMN_MAX_WIDTH ) {
                fprintf( stdout, "ERROR: column name too long!\n");
                fflush( stdout );
                return FALSE;
            }
        } else {
            if( istr == 0 ) {
                fprintf( stdout, "ERROR: column name empty!\n");
                fflush( stdout );
                return FALSE;
            } else {
                data->col_num ++;
                istr = 0;
            }
        }
    }
    
    if ( col_num > 0 ) {
        if ( col_num > data->col_num ) {
            fprintf( stdout, "ERROR: column number out of range!\n" );
            fflush( stdout );
            return FALSE;
        } else {
            data->col_num = col_num;
        }
    }
    
    if ( !data->col_num ) {
        fprintf( stdout, "ERROR: no columns found!\n" );
        fflush( stdout );
        return FALSE;
    }
    
    // allocate for head names
    data->variables = (char**)malloc( sizeof(char*) * data->col_num );
    
    // second scan to get head names
    fseek( fp, 0, SEEK_SET );
    istr = 0;
    while ( TRUE ) {
        c = fgetc( fp );
        if ( c == row_delim ) {
            var_name = (char*)malloc( sizeof(char) * ( istr + 1 ) );
            memcpy( var_name, str, istr );
            var_name[istr] = 0; // close string
            data->variables[icol++] = var_name;
            
            fprintf( stdout, "INFO: file head names read!\n" );
            fflush( stdout );
            break;
        }
        if ( c != col_delim ) {
            str[istr++] = c;
        } else {
            var_name = (char*)malloc( sizeof(char) * ( istr + 1 ) );
            memcpy( var_name, str, istr );
            var_name[istr] = 0; // close string
            data->variables[icol++] = var_name;
            istr = 0;
        }
    }
    
    // continue the second scan to get row number
    while ( !feof( fp ) ) {
        c = fgetc( fp );
        if ( c == row_delim ) {
            data->row_num ++;
        }
    }
    /*// last row accounted
    if ( c != row_delim ) {
        data->row_num ++;
    }*/
    if ( row_num > 0 ) {
        if ( row_num > data->row_num ) {
            fprintf( stdout, "ERROR: row number out of range!\n" );
            fflush( stdout );
            return FALSE;
        } else {
            data->row_num = row_num;
        }
    }
    
    if ( !data->row_num ) {
        fprintf( stdout, "ERROR: row number cannot be 0!\n" );
        fflush( stdout );
        return FALSE;
    } else {
        fprintf( stdout, "INFO: row number read as : %d.\n", data->row_num );
        fflush( stdout );
    }
    
    // allocate for value matrix
    data->values = (char***)malloc( sizeof(char**) * data->col_num );
    for ( i = 0; i < data->col_num; i++ ) {
        data->values[i] = (char**)malloc( sizeof(char*) * data->row_num );
    }
    
    // run third scan to fill matrix
    istr = 0;
    irow = 0;
    icol = 0;
    skip_to_next_row = FALSE;
    fseek( fp, 0, SEEK_SET );
    
    // move to second row
    c = fgetc( fp );
    while ( c != row_delim ) {
        c = fgetc( fp );
    }
    
	while ( !feof( fp ) ) {
		c = fgetc( fp );
        if( skip_to_next_row && ( c != row_delim ) ) {
            continue;
        }
        if ( c == row_delim ) {
            var_value = (char*)malloc( sizeof(char) * ( istr + 1 ) );
            memcpy( var_value, str, istr );
            var_value[istr] = 0; // close string
            data->values[icol++][irow++] = var_value;
            if ( icol < data->col_num ) {
                fprintf( stdout, "ERROR: column missing at row: %d!\n", irow );
                fprintf( stdout, "ERROR: only %d columns found in this row.\n", icol );
                fflush( stdout );
                return FALSE;
            }
            icol = 0;
            istr = 0;
            skip_to_next_row = FALSE;
        } else {
            if ( c == col_delim ) {
                var_value = (char*)malloc( sizeof(char) * ( istr + 1 ) );
                memcpy( var_value, str, istr );
                var_value[istr] = 0; // close string
                data->values[icol++][irow] = var_value;
                istr = 0;
            } else {
                str[istr++] = c;
                if ( istr >= COLUMN_MAX_WIDTH ) {
                    fprintf( stdout, "ERROR: column value string too long!\n" );
                    fflush( stdout );
                    return FALSE;
                }
            }
        }
        // check if meets column number
        if ( icol == data->col_num ) {
            irow ++;
            skip_to_next_row = TRUE;
        }
        // check if meets row number
        if ( irow == data->row_num ) {
            break;
        }
	}
    /*
    // save last value
    if ( feof( fp ) ) { // last row in file
        if ( !skip_to_next_row ) { // last char is not row delimiter
            if ( c != col_delim ) { // last char is not a col dilimiter
                var_value = (char*)malloc( sizeof(char) * ( istr + 1 ) );
                memcpy( var_value, str, istr );
                var_value[istr] = 0; // close string
                data->values[icol++][irow++] = var_value;
            }
        }
    }*/
    
    if ( irow == data->row_num ) {
        fprintf( stdout, "INFO: matrix value read OK!\n" );
        fflush( stdout );
    } else {
        fprintf( stdout, "ERROR: matrix dimension not match!\n" );
        fprintf( stdout, "matrix read row number: %d.", irow );
        fflush( stdout );
        return FALSE;
    }
    
    // print header
    fprintf( stdout, "VARIABLES:\n" );
    for ( i=0; i<data->col_num; i++ ) {
        fprintf( stdout, "[%s]\t", data->variables[i] );
    }
    fprintf( stdout, "\n" );
    fflush( stdout );
    
    // convert to make extended data
    if ( !value_types ) {
        fprintf( stdout, "ERROR: value types not set!\n" );
        fflush( stdout );
        return FALSE;
    }
    if ( !variable_types ) {
        fprintf( stdout, "ERROR: variable types not set!\n" );
        fflush( stdout );
        return FALSE;
    }
    if ( !variable_roles ) {
        fprintf( stdout, "ERROR: variable roles not set!\n" );
        fflush( stdout );
        return FALSE;
    }
    
    for ( i=0; i<data->col_num; i++ ) {
        t_val = value_types[i];
        t_var = variable_types[i];
        t_role = variable_roles[i];
        
        for ( j=0; j<data->row_num; j++ ) {
            
        }
    }
    
	fclose( fp );
	
	return TRUE;
}
