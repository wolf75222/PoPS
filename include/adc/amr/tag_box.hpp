#pragma once

#include <adc/mesh/box2d.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

// TagBox : grille dense de marqueurs (0/1) sur une region, entree du clustering
// Berger-Rigoutsos. Stockage i-rapide. Pour MPI plus tard, les tags repartis
// seront rassembles sur cette grille avant clustering (le clustering est bon
// marche face au reste).

namespace adc {

struct TagBox {
  Box2D box{};
  std::vector<char> t{};

  TagBox() = default;
  explicit TagBox(const Box2D& b)
      : box(b),
        t(static_cast<std::size_t>(std::max<long>(0, b.num_cells())), 0) {}

  char& operator()(int i, int j) { return t[idx(i, j)]; }
  char operator()(int i, int j) const { return t[idx(i, j)]; }
  bool tagged(int i, int j) const {
    return box.contains(i, j) && t[idx(i, j)] != 0;
  }

  long count() const {
    long c = 0;
    for (char x : t) c += x;
    return c;
  }

 private:
  std::size_t idx(int i, int j) const {
    return static_cast<std::size_t>(j - box.lo[1]) * box.nx() + (i - box.lo[0]);
  }
};

// UNION (OU logique cellule a cellule) de plusieurs TagBox partageant EXACTEMENT la meme box.
// Brique du regrid d'union des tags multi-blocs (docs/AMR_REGRID_UNION_TAGS_DESIGN.md etape R3) :
// chaque bloc + le critere phi produisent une TagBox sur le MEME domaine parent, l'union est leur
// OU bit a bit. Sans dependance physique (quelques lignes). Liste vide -> TagBox vide (box par
// defaut). Une box discordante leve (les TagBox d'union DOIVENT couvrir le meme parent, sinon
// l'indexation lineaire melangerait deux geometries). Stockage i-rapide -> simple |= sur le buffer.
inline TagBox tag_union(const std::vector<TagBox>& parts) {
  if (parts.empty()) return TagBox{};
  TagBox out(parts[0].box);
  for (const TagBox& tb : parts) {
    if (tb.box.lo[0] != out.box.lo[0] || tb.box.lo[1] != out.box.lo[1] ||
        tb.box.hi[0] != out.box.hi[0] || tb.box.hi[1] != out.box.hi[1])
      throw std::runtime_error(
          "tag_union : toutes les TagBox doivent partager EXACTEMENT la meme box (meme domaine "
          "parent) pour l'union cellule a cellule");
    const std::size_t n = std::min(out.t.size(), tb.t.size());
    for (std::size_t k = 0; k < n; ++k) out.t[k] |= tb.t[k];
  }
  return out;
}

}  // namespace adc
