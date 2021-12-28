#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <queue>
#include <vector>
#include <utility>
#include <iostream>

template <typename T>
class Vecs {
  public:
    Vecs(char* filename, size_t num_dims){
      bytesPerElement = sizeof(T);
      d = num_dims;

      filesize = getFilesize(filename);
      n = filesize / (d*bytesPerElement);
      int fd = open(filename, O_RDONLY, 0);
      assert(fd != -1);
      // https://techoverflow.net/2013/08/21/a-simple-mmap-readonly-example/
      data = (T*)mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
      assert(data != MAP_FAILED);
      //TODO: madvise?

    }

    ~Vecs(){
      munmap(data, filesize);
      close(fd);
    }

    size_t size() {
      return n;
    }

    size_t vectorSizeBytes() {
      return d*bytesPerElement;
    }

    size_t numDimensions() {
      return d;
    }

    void debugPrintVector(size_t i){
      T* p = get(i);

      printf("debug: sizeof(T): %lu\n", sizeof(T));

      for(size_t j = 0; j < d; j++){
        if(std::is_same<T, unsigned char>::value){
          std::cout<<int(p[j]);
        }
        else{
          std::cout<<p[j];
        }

        if(j<d-1){
          std::cout<<", ";
        }
      }
      std::cout<<std::endl;
    }

    T* get(size_t i){
      return data + i*d;
    };

  private:
    size_t n;
    size_t d;
    size_t bytesPerElement;
    size_t filesize;
    T* data;
    int fd;

    size_t getFilesize(const char* filename) {
      struct stat st;
      stat(filename, &st);
      return st.st_size;
    }
};

template<typename T>
class QueryRunner {

  private:
    bool compare_distance(std::pair<T, size_t> x, std::pair<T, size_t> y){
      return x.first < y.first;
    }

    struct CompareDistance{
      bool operator()(std::pair<T, size_t> x, std::pair<T, size_t> y){
        return x.first < y.first;
      }
    };

  public:
    typedef std::pair<T,size_t> dist_ix;
    typedef std::priority_queue<dist_ix, std::vector<dist_ix>, CompareDistance> dist_ix_pq;

    QueryRunner(size_t num_dims, Vecs<T>* query_set, Vecs<T>* data_base, size_t k_nearest){
      querySet = query_set;
      database = data_base;
      k = k_nearest;
      d = database->numDimensions();

      kNearest = std::vector<dist_ix_pq>();

      for(size_t i = 0; i < querySet->size(); i++){
        kNearest.emplace_back(dist_ix_pq());
      }

    };

    void search_over(size_t start_inclusive, size_t end_exclusive){
      for(size_t i = start_inclusive; i < end_exclusive; i++){
        for(size_t q = 0; q < querySet->size(); q++){
          try_insert(q, i);
        }
      }
    }

    void try_insert(size_t q, size_t i){
      T dist = distance(q, i);

      kNearest[q].emplace(dist, i);

      while(kNearest[q].size() > k){
        kNearest[q].pop();
      }
    }

    // merge self with another QueryRunner. The other QueryRunner will be
    // mutated to have empty kNearest priority_queues and should be discarded.
    void merge_with(QueryRunner<T> other){
      for(size_t q = 0; q < querySet->size(); q++){
        while(other.kNearest[q].size()){
          kNearest[q].push(other.kNearest[q].pop());
        }
        while(kNearest[q].size() > k){
          kNearest[q].pop();
        }
      }
    }

  private:
    Vecs<T>* querySet;
    Vecs<T>* database;
    size_t d;
    size_t k;
    // kNearest[q] stores the k-nearest neighbors of q found so far, as pairs of
    // distance, index (into database).
    std::vector<dist_ix_pq> kNearest;

    // Returns the distance from querySet[q] to database[i].
    // Not the actual distance because no sqrt call is made. Use the result to
    // compare magnitudes only.
    T distance(size_t q, size_t i){
      T ret = 0;
      T* qPtr = querySet->get(q);
      T* iPtr = database->get(i);

      for(size_t j = 0; j < d; j++){
        T diff = qPtr[j] - iPtr[j];
        T sq_diff = diff * diff;
        ret += sq_diff;
      }

      return ret;
    }


};

template <typename T>
void run_search(size_t num_threads, size_t num_dims, size_t k, char* database_filename, char* query_filename){

  Vecs<T> querySet = Vecs<T>(query_filename, num_dims);
  Vecs<T> database = Vecs<T>(database_filename, num_dims);

  printf("Running search for the %zu nearest neighbors of %zu query points on a database of %zu points\n", k, querySet.size(), database.size());

  QueryRunner<T> qr = QueryRunner<T>(num_dims, &querySet, &database, k);
  qr.search_over(0, database.size());

  return;
}

int main(int argc, char *argv[]){

  if(argc != 7){
    fprintf(stderr, "Usage: ./search [f|i|b] num_threads num_dims k base.bvecs_array query.bvecs_array");
  }

  char elemType = argv[1][0];
  int num_threads = strtol(argv[2], NULL, 10);
  size_t num_dims = strtol(argv[3], NULL, 10);
  size_t k = strtol(argv[4], NULL, 10);
  char* database_filename = argv[5];
  char* query_filename = argv[6];

  switch(elemType) {
    case 'f':
      run_search<float>(num_threads, num_dims, k, database_filename, query_filename);
      break;
    case 'i':
      run_search<int>(num_threads, num_dims, k, database_filename, query_filename);
    case 'b':
      run_search<unsigned char>(num_threads, num_dims, k, database_filename, query_filename);
  }


  return 0;
}
