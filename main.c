#include "read.h"

int main( int argc, const char ** argv ) {
	odata od;
	/*if ( argc < 2 ) {
		fprintf( stdout, "input data file path required!\n" );
		return 1;
	}
    if ( read( &od, argv[1], '\t', '\n', 0, 0, 0 ) ) {
        fprintf( stdout, "failed to read file!\n" );
		return 1;
    }*/

    if ( !read( &od, "_2G_FILTERED.CSV", ',', '\n', 0, 0, NULL, NULL, NULL, FALSE, FALSE ) ) {
        fprintf( stdout, "failed to read file!\n" );
        fflush( stdout );
		return 1;
    }

	return 0;
}
