#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*pops_python_main)(int, char**);

static int load_optional_global(const char* variable, const char* description)
{
  const char* library = getenv(variable);
  if (library == NULL || library[0] == '\0')
  {
    return 0;
  }
  if (dlopen(library, RTLD_NOW | RTLD_GLOBAL) == NULL)
  {
    fprintf(stderr, "cannot preload %s: %s\n", description, dlerror());
    return 126;
  }
  return 0;
}

int main(int argc, char** argv)
{
  if (load_optional_global("POPS_ACTIVE_MPI_LIBRARY", "active PoPS MPI") != 0 ||
    load_optional_global("POPS_ACTIVE_MPICXX_LIBRARY", "active PoPS MPI C++") != 0 ||
    load_optional_global("POPS_ACTIVE_HDF5_LIBRARY", "active PoPS parallel HDF5") != 0)
  {
    return 126;
  }
  const char* library = getenv("POPS_PARAVIEW_LIBPYTHON");
  if (library == NULL || library[0] == '\0')
  {
    fputs("POPS_PARAVIEW_LIBPYTHON is required\n", stderr);
    return 126;
  }
  void* handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
  if (handle == NULL)
  {
    fprintf(stderr, "cannot load ParaView Python: %s\n", dlerror());
    return 126;
  }
  dlerror();
  void* symbol = dlsym(handle, "Py_BytesMain");
  const char* error = dlerror();
  if (error != NULL)
  {
    fprintf(stderr, "ParaView Python does not export Py_BytesMain: %s\n", error);
    return 126;
  }
  pops_python_main run = NULL;
  if (sizeof(run) != sizeof(symbol))
  {
    fputs("unsupported function-pointer ABI for ParaView Python\n", stderr);
    return 126;
  }
  memcpy(&run, &symbol, sizeof(run));
  return run(argc, argv);
}
