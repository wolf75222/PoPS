#pragma once

// Le seam parallele (rang / nombre de rangs). Rang unique pour l'instant ;
// l'implementation MPI ne changera que ce fichier (MPI_Comm_rank /
// MPI_Comm_size), le reste du code passe par my_rank() / n_ranks().

namespace adc {

inline int my_rank() { return 0; }
inline int n_ranks() { return 1; }

}  // namespace adc
