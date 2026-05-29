#pragma once

// DataWriter HDF5 : sortie distribuee a l'echelle, sans gather. Equivalent maison
// du DataWriter HDF5/NetCDF parallele (vs notre dump texte + gather global).
//
// Un MultiFab est ecrit dans un dataset HDF5 [ny][nx] couvrant le domaine GLOBAL.
// Chaque rang ecrit SES boites locales via une hyperslab (selection d'un
// sous-rectangle du dataset). En parallele (HDF5 compile avec MPI-IO) les ecritures
// vont vers des regions DISJOINTES et sont independantes (H5FD_MPIO_INDEPENDENT :
// pas de synchro collective, donc les rangs peuvent avoir des nombres de boites
// differents). En serie c'est une simple ecriture. Aucune donnee ne transite par
// le rang 0 : le fichier est ecrit en place, scalable.
//
// Gate ADC_HAS_HDF5 (option CMake ADC_USE_HDF5). Sans HDF5 ce header est vide.

#ifdef ADC_HAS_HDF5

#include <hdf5.h>

#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <string>
#include <vector>

namespace adc {

// Ecrit la composante `comp` de `mf` dans `filename:dataset`, dataset [ny][nx]
// (ny lignes = direction j, nx colonnes = direction i) du domaine global.
inline void write_hdf5(const MultiFab& mf, int nx, int ny,
                       const std::string& filename,
                       const std::string& dataset = "field", int comp = 0) {
  device_fence();  // GPU : donnees ecrites par un kernel -> barriere avant lecture hote

  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#if defined(ADC_HAS_MPI) && defined(H5_HAVE_PARALLEL)
  const bool par = n_ranks() > 1;
  if (par) H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif
  hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  H5Pclose(fapl);

  const hsize_t dims[2] = {static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
  hid_t filespace = H5Screate_simple(2, dims, nullptr);
  hid_t dset = H5Dcreate2(file, dataset.c_str(), H5T_NATIVE_DOUBLE, filespace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hid_t xfer = H5P_DEFAULT;
#if defined(ADC_HAS_MPI) && defined(H5_HAVE_PARALLEL)
  if (par) {
    xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_INDEPENDENT);  // regions disjointes
  }
#endif

  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    const Box2D b = mf.box(li);
    const hsize_t ni = static_cast<hsize_t>(b.hi[0] - b.lo[0] + 1);
    const hsize_t nj = static_cast<hsize_t>(b.hi[1] - b.lo[1] + 1);
    std::vector<double> buf(ni * nj);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        buf[static_cast<std::size_t>(j - b.lo[1]) * ni + (i - b.lo[0])] =
            static_cast<double>(a(i, j, comp));

    const hsize_t offset[2] = {static_cast<hsize_t>(b.lo[1]),
                               static_cast<hsize_t>(b.lo[0])};
    const hsize_t count[2] = {nj, ni};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count,
                        nullptr);
    hid_t memspace = H5Screate_simple(2, count, nullptr);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer, buf.data());
    H5Sclose(memspace);
  }

#if defined(ADC_HAS_MPI) && defined(H5_HAVE_PARALLEL)
  if (par) H5Pclose(xfer);
#endif
  H5Dclose(dset);
  H5Sclose(filespace);
  H5Fclose(file);
}

}  // namespace adc

#endif  // ADC_HAS_HDF5
