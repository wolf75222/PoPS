extern int mpi_expected_value(void);
extern int mpi_other_value(void);
int hdf5_dual_fixture_value(void) { return mpi_expected_value() + mpi_other_value(); }
