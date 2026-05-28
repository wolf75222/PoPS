#pragma once

// Le seam parallele. Sans ADC_HAS_MPI : rang unique (serie). Avec ADC_HAS_MPI :
// MPI_Comm_rank/size + collectives sur MPI_COMM_WORLD. Tout le reste du code
// passe par my_rank() / n_ranks() / all_reduce_* et ignore le backend.
//
// Robustesse : meme compile avec MPI, si MPI n'est pas initialise (ex. un test
// serie linke contre adc), my_rank() rend 0 et n_ranks() rend 1. Il faut donc
// appeler comm_init() au debut de main() pour les runs reellement distribues.

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

namespace adc {

#ifdef ADC_HAS_MPI

inline bool comm_active() {
  int inited = 0, fin = 0;
  MPI_Initialized(&inited);
  MPI_Finalized(&fin);
  return inited && !fin;
}

inline void comm_init(int* argc = nullptr, char*** argv = nullptr) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) MPI_Init(argc, argv);
}

inline void comm_finalize() {
  int fin = 0;
  MPI_Finalized(&fin);
  if (!fin) MPI_Finalize();
}

inline int my_rank() {
  if (!comm_active()) return 0;
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

inline int n_ranks() {
  if (!comm_active()) return 1;
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

inline void barrier() {
  if (comm_active()) MPI_Barrier(MPI_COMM_WORLD);
}

inline double all_reduce_sum(double x) {
  if (!comm_active()) return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

inline double all_reduce_max(double x) {
  if (!comm_active()) return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return r;
}

inline long all_reduce_sum(long x) {
  if (!comm_active()) return x;
  long r = x;
  MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

#else  // ----- serie -----

inline bool comm_active() { return false; }
inline void comm_init(int* = nullptr, char*** = nullptr) {}
inline void comm_finalize() {}
inline int my_rank() { return 0; }
inline int n_ranks() { return 1; }
inline void barrier() {}
inline double all_reduce_sum(double x) { return x; }
inline double all_reduce_max(double x) { return x; }
inline long all_reduce_sum(long x) { return x; }

#endif

}  // namespace adc
