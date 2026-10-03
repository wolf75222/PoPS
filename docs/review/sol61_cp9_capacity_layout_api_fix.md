# CP9 capacity actual Layout API correction

Authentic build22 job732222 failed in checkpoint_state_carriers_capacity: MultiFab::layout returns mesh::BoxArray, which exposes size/operator[] and boxes(), but no begin/end. The vector-based earlier host adapter masked this API incompatibility.

Only the production traversal changes: bind the existing global layout by const reference, visit every patch index, then use the identical box.numPts, uint64 accumulation overflow guard and maximum across every State block. No Layout API extension, capacity padding, model/State special case, ABI8 or POPSCAR1/CP9 wire change.

The new host test extracts the actual production count fragment and compiles it against the actual BoxArray/Box headers, dimensions1/2/3. Independently expected two disjoint negatively/positively indexed boxes and empty layout counts are checked. Parent99c actual fragment fails to compile in all three dimensions (3 RED,1.42s); fixed fragment plus existing capacity cohort passes12tests6.50s. This proves real header API compatibility for that fragment, not the full System TU, Native MPI/Kokkos/GPU or runtime checkpoint. ROOT must rebuild.

Evidence: /Users/romaindespoulain/dev/tmp/sol61-cp9-layout-api-fix-20261003. Baseline99c805690cbca380abe5d08b157aafaa1959cb3e; Root Native checkout/ENV/raw failed build were not modified.
