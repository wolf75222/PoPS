/* MPI profiling-interface (PMPI) trace of collective calls, for N9 (ROM-90, SV20).
 *
 * Loaded with LD_PRELOAD in front of the MPI library, it records one entry per collective call
 * (operation, element count, element size, reduction operator) and counts point-to-point calls.
 * It changes no PoPS code path: every wrapper forwards to its PMPI_ entry point unchanged.
 * measure_cg_collectives.py reads the trace through ctypes. With POPS_N9_TRACE_FRAMES=<depth>
 * set, each entry also keeps its first <depth> return addresses (glibc backtrace), which the
 * script resolves to the calling functions to attribute every collective to its source. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  POPS_N9_ALLREDUCE = 1,
  POPS_N9_ALLGATHER,
  POPS_N9_ALLGATHERV,
  POPS_N9_BCAST,
  POPS_N9_BARRIER,
  POPS_N9_REDUCE,
  POPS_N9_GATHER,
  POPS_N9_GATHERV,
  POPS_N9_ALLTOALL,
  POPS_N9_ALLTOALLV,
  POPS_N9_SCATTER,
  POPS_N9_SCATTERV,
  POPS_N9_REDUCE_SCATTER_BLOCK,
  POPS_N9_SCAN,
  POPS_N9_EXSCAN,
  POPS_N9_IALLREDUCE,
  POPS_N9_IBARRIER,
  POPS_N9_IBCAST,
  POPS_N9_COMM_DUP,
  POPS_N9_COMM_SPLIT
};

typedef struct {
  int32_t kind;
  int32_t count;
  int32_t element_bytes;
  int32_t op;
} pops_n9_entry;

static pops_n9_entry* trace = NULL;
static int64_t trace_size = 0;
static int64_t trace_capacity = 0;
static int64_t trace_dropped = 0;
static int64_t point_to_point = 0;
enum { POPS_N9_MAX_FRAMES = 24 };
static int frame_depth = -1; /* -1: not read yet, 0: frames off */
static void** frames = NULL; /* frame_depth addresses per entry, NULL-padded */
static int64_t frames_capacity = 0;

static int read_frame_depth(void) {
  if (frame_depth < 0) {
    const char* value = getenv("POPS_N9_TRACE_FRAMES");
    const int depth = value == NULL ? 0 : atoi(value);
    frame_depth = depth < 0 ? 0 : (depth > POPS_N9_MAX_FRAMES ? POPS_N9_MAX_FRAMES : depth);
  }
  return frame_depth;
}

static void record_frames(int64_t index) {
  const int depth = read_frame_depth();
  if (depth == 0)
    return;
  if (index >= frames_capacity) {
    const int64_t next = frames_capacity == 0 ? 65536 : 2 * frames_capacity;
    void** grown = (void**)realloc(frames, (size_t)next * (size_t)depth * sizeof(void*));
    if (grown == NULL) {
      ++trace_dropped;
      return;
    }
    frames = grown;
    frames_capacity = next;
  }
  void* stack[POPS_N9_MAX_FRAMES + 2];
  const int found = backtrace(stack, depth + 2);
  void** row = frames + index * depth;
  for (int i = 0; i < depth; ++i)
    row[i] = i + 2 < found ? stack[i + 2] : NULL; /* skip shim */
}

static int op_code(MPI_Op op) {
  if (op == MPI_SUM)
    return 1;
  if (op == MPI_MAX)
    return 2;
  if (op == MPI_MIN)
    return 3;
  if (op == MPI_LOR)
    return 4;
  if (op == MPI_BOR)
    return 5;
  if (op == MPI_LAND)
    return 6;
  if (op == MPI_BAND)
    return 7;
  if (op == MPI_MAXLOC)
    return 8;
  if (op == MPI_MINLOC)
    return 9;
  return 0; /* user-defined */
}

static int type_bytes(MPI_Datatype type) {
  int bytes = 0;
  if (type == MPI_DATATYPE_NULL || PMPI_Type_size(type, &bytes) != MPI_SUCCESS)
    return -1;
  return bytes;
}

static void record(int kind, int count, MPI_Datatype type, int op) {
  if (trace_size == trace_capacity) {
    const int64_t next = trace_capacity == 0 ? 65536 : 2 * trace_capacity;
    if (next > ((int64_t)1 << 26)) {
      ++trace_dropped;
      return;
    }
    pops_n9_entry* grown = (pops_n9_entry*)realloc(trace, (size_t)next * sizeof(pops_n9_entry));
    if (grown == NULL) {
      ++trace_dropped;
      return;
    }
    trace = grown;
    trace_capacity = next;
  }
  trace[trace_size].kind = kind;
  trace[trace_size].count = count;
  trace[trace_size].element_bytes = type_bytes(type);
  trace[trace_size].op = op;
  record_frames(trace_size);
  ++trace_size;
}

/* Query interface, resolved by ctypes from the preloaded library. */
int64_t pops_n9_trace_size(void) {
  return trace_size;
}
int64_t pops_n9_trace_dropped(void) {
  return trace_dropped;
}
int64_t pops_n9_point_to_point(void) {
  return point_to_point;
}
void pops_n9_trace_reset(void) {
  trace_size = 0;
  trace_dropped = 0;
  point_to_point = 0;
}
int64_t pops_n9_trace_copy(int64_t first, int64_t count, int32_t* out) {
  int64_t copied = 0;
  for (int64_t i = first; i < trace_size && copied < count; ++i, ++copied) {
    out[4 * copied + 0] = trace[i].kind;
    out[4 * copied + 1] = trace[i].count;
    out[4 * copied + 2] = trace[i].element_bytes;
    out[4 * copied + 3] = trace[i].op;
  }
  return copied;
}
int pops_n9_trace_frame_depth(void) {
  return read_frame_depth();
}
/* Copies the return addresses of entries [first, first + count), frame_depth per entry. */
int64_t pops_n9_trace_frames(int64_t first, int64_t count, uint64_t* out) {
  const int depth = read_frame_depth();
  int64_t copied = 0;
  if (depth == 0 || frames == NULL)
    return 0;
  for (int64_t i = first; i < trace_size && copied < count; ++i, ++copied)
    for (int f = 0; f < depth; ++f)
      out[copied * depth + f] = (uint64_t)(uintptr_t)frames[i * depth + f];
  return copied;
}
/* Object file and load base of one address, from the dynamic loader (dladdr). */
int pops_n9_resolve(uint64_t address, char* path, int path_size, uint64_t* base) {
  Dl_info info;
  memset(&info, 0, sizeof(info));
  if (dladdr((void*)(uintptr_t)address, &info) == 0 || info.dli_fname == NULL)
    return 0;
  strncpy(path, info.dli_fname, (size_t)path_size - 1);
  path[path_size - 1] = '\0';
  *base = (uint64_t)(uintptr_t)info.dli_fbase;
  return 1;
}

int MPI_Allreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
  record(POPS_N9_ALLREDUCE, n, t, op_code(o));
  return PMPI_Allreduce(s, r, n, t, o, c);
}
int MPI_Iallreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c,
                   MPI_Request* q) {
  record(POPS_N9_IALLREDUCE, n, t, op_code(o));
  return PMPI_Iallreduce(s, r, n, t, o, c, q);
}
int MPI_Reduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, int root, MPI_Comm c) {
  record(POPS_N9_REDUCE, n, t, op_code(o));
  return PMPI_Reduce(s, r, n, t, o, root, c);
}
int MPI_Allgather(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt,
                  MPI_Comm c) {
  record(POPS_N9_ALLGATHER, sn, st, -1);
  return PMPI_Allgather(s, sn, st, r, rn, rt, c);
}
int MPI_Allgatherv(const void* s, int sn, MPI_Datatype st, void* r, const int* rn,
                   const int* displs, MPI_Datatype rt, MPI_Comm c) {
  record(POPS_N9_ALLGATHERV, sn, st, -1);
  return PMPI_Allgatherv(s, sn, st, r, rn, displs, rt, c);
}
int MPI_Gather(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, int root,
               MPI_Comm c) {
  record(POPS_N9_GATHER, sn, st, -1);
  return PMPI_Gather(s, sn, st, r, rn, rt, root, c);
}
int MPI_Gatherv(const void* s, int sn, MPI_Datatype st, void* r, const int* rn, const int* displs,
                MPI_Datatype rt, int root, MPI_Comm c) {
  record(POPS_N9_GATHERV, sn, st, -1);
  return PMPI_Gatherv(s, sn, st, r, rn, displs, rt, root, c);
}
int MPI_Bcast(void* b, int n, MPI_Datatype t, int root, MPI_Comm c) {
  record(POPS_N9_BCAST, n, t, -1);
  return PMPI_Bcast(b, n, t, root, c);
}
int MPI_Ibcast(void* b, int n, MPI_Datatype t, int root, MPI_Comm c, MPI_Request* q) {
  record(POPS_N9_IBCAST, n, t, -1);
  return PMPI_Ibcast(b, n, t, root, c, q);
}
int MPI_Barrier(MPI_Comm c) {
  record(POPS_N9_BARRIER, 0, MPI_DATATYPE_NULL, -1);
  return PMPI_Barrier(c);
}
int MPI_Ibarrier(MPI_Comm c, MPI_Request* q) {
  record(POPS_N9_IBARRIER, 0, MPI_DATATYPE_NULL, -1);
  return PMPI_Ibarrier(c, q);
}
int MPI_Alltoall(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt,
                 MPI_Comm c) {
  record(POPS_N9_ALLTOALL, sn, st, -1);
  return PMPI_Alltoall(s, sn, st, r, rn, rt, c);
}
int MPI_Alltoallv(const void* s, const int* sn, const int* sd, MPI_Datatype st, void* r,
                  const int* rn, const int* rd, MPI_Datatype rt, MPI_Comm c) {
  record(POPS_N9_ALLTOALLV, 0, st, -1);
  return PMPI_Alltoallv(s, sn, sd, st, r, rn, rd, rt, c);
}
int MPI_Scatter(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, int root,
                MPI_Comm c) {
  record(POPS_N9_SCATTER, rn, rt, -1);
  return PMPI_Scatter(s, sn, st, r, rn, rt, root, c);
}
int MPI_Scatterv(const void* s, const int* sn, const int* displs, MPI_Datatype st, void* r, int rn,
                 MPI_Datatype rt, int root, MPI_Comm c) {
  record(POPS_N9_SCATTERV, rn, rt, -1);
  return PMPI_Scatterv(s, sn, displs, st, r, rn, rt, root, c);
}
int MPI_Reduce_scatter_block(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
  record(POPS_N9_REDUCE_SCATTER_BLOCK, n, t, op_code(o));
  return PMPI_Reduce_scatter_block(s, r, n, t, o, c);
}
int MPI_Scan(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
  record(POPS_N9_SCAN, n, t, op_code(o));
  return PMPI_Scan(s, r, n, t, o, c);
}
int MPI_Exscan(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
  record(POPS_N9_EXSCAN, n, t, op_code(o));
  return PMPI_Exscan(s, r, n, t, o, c);
}
int MPI_Comm_dup(MPI_Comm c, MPI_Comm* out) {
  record(POPS_N9_COMM_DUP, 0, MPI_DATATYPE_NULL, -1);
  return PMPI_Comm_dup(c, out);
}
int MPI_Comm_split(MPI_Comm c, int color, int key, MPI_Comm* out) {
  record(POPS_N9_COMM_SPLIT, 0, MPI_DATATYPE_NULL, -1);
  return PMPI_Comm_split(c, color, key, out);
}

int MPI_Isend(const void* b, int n, MPI_Datatype t, int d, int tag, MPI_Comm c, MPI_Request* q) {
  ++point_to_point;
  return PMPI_Isend(b, n, t, d, tag, c, q);
}
int MPI_Irecv(void* b, int n, MPI_Datatype t, int s, int tag, MPI_Comm c, MPI_Request* q) {
  ++point_to_point;
  return PMPI_Irecv(b, n, t, s, tag, c, q);
}
int MPI_Send(const void* b, int n, MPI_Datatype t, int d, int tag, MPI_Comm c) {
  ++point_to_point;
  return PMPI_Send(b, n, t, d, tag, c);
}
int MPI_Recv(void* b, int n, MPI_Datatype t, int s, int tag, MPI_Comm c, MPI_Status* st) {
  ++point_to_point;
  return PMPI_Recv(b, n, t, s, tag, c, st);
}
int MPI_Sendrecv(const void* sb, int sn, MPI_Datatype st, int d, int stag, void* rb, int rn,
                 MPI_Datatype rt, int s, int rtag, MPI_Comm c, MPI_Status* status) {
  ++point_to_point;
  return PMPI_Sendrecv(sb, sn, st, d, stag, rb, rn, rt, s, rtag, c, status);
}
