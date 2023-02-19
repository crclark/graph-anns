An example of running this library on the [TexMex corpus](http://corpus-texmex.irisa.fr/).

To use, you first need to preprocess the texmex data so that it can be mmaped directly as a C-style array. This code removes the redundant array length at the beginning of each vector, and writes the vectors directly to stdout.

```c
/*
Converts a file in the format specified here: http://corpus-texmex.irisa.fr/
into a simple 2d-array of number_of_vectors*number_of_dimensions. This allows us
to simply mmap the output of this program, cast it as an appropriately-sized array,
and use it directly with good performance.

The program reads from stdin and writes to stdout (as binary).

Example 1: pv -pteIrabT bigann_learn.bvecs | ./convert 1 128 > bigann_learn.bvecs_array

*/

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

  if(argc != 3) {
    printf("Usage: cat foo.fvecs | ./convert bytes_per_coord dimensions\n"
           "\n"
           "Example 1: pv -pteIrabT bigann_learn.bvecs | ./convert 1 128 > bigann_learn.bvecs_array\n"
           "\n"
           "Example 2: pv -pteIrabT siftsmall_base.fvecs | ./convert 4 128 > siftsmall_base.fvecs_array\n");
    return 1;
  }

  size_t bytes_per_coord = strtol(argv[1], NULL, 10);
  size_t dimensions = strtol(argv[2], NULL, 10);

  size_t buffer_size = bytes_per_coord * dimensions;

  // set up our streams in binary mode
  freopen(NULL, "rb", stdin);
  freopen(NULL, "wb", stdout);

  void* buffer = malloc(buffer_size);

  void* skip_buffer = malloc(4);

  while(1){

    size_t skip_items_read = fread(skip_buffer, 4, 1, stdin);
    if(skip_items_read == 0){
      break; // reached EOF
    }

    // send the next vector to stdout.
    size_t items_read = fread(buffer, bytes_per_coord, dimensions, stdin);
    // we should always read exactly the number of dimensions. If it's zero, we
    // are done.
    if (items_read == 0){
      break;
    }
    else if(items_read != dimensions){
      fprintf(stderr, "read error\n");
      return 1;
    }


    size_t items_written = fwrite(buffer, bytes_per_coord, dimensions, stdout);
    if (items_written != dimensions){
      fprintf(stderr, "write error: wrote %zu items\n", items_written);
      fprintf(stderr, "error number is %d\n", ferror(stdout));
      fprintf(stderr, "feof: %d\n", feof(stdout));
      return 1;
    }
  }

  return 0;
}
```
