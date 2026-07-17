#pragma once

#include <pops/core/state/state.hpp>  // kAuxBaseComps (default width of the shared aux channel)
#include <pops/parallel/comm.hpp>     // n_ranks (round-robin distribution mapping)
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>        // Geometry, PolarGeometry
#include <pops/mesh/storage/multifab.hpp>         // MultiFab
#include <pops/mesh/boundary/physical_bc.hpp>     // BCRec, BCType, Periodicity
#include <pops/mesh/index/box2d.hpp>              // Box2D
#include <pops/runtime/context/grid_context.hpp>  // GeometryMode, detail::DiscDomain
#include <pops/runtime/system.hpp>  // SystemConfig (the geometry/layout source; system.hpp does NOT include this header, so no cycle)

#include <cstddef>
#include <utility>
#include <vector>

/// @file
/// @brief The GEOMETRY / LAYOUT registry of a System (ADC-578).
///
/// Extracted from the geometry + mesh-layout members that lived inline on `System::Impl`: the index
/// geometry (Cartesian `geom` or polar `pgeom_`), the box array / distribution mapping (`ba` / `dm`),
/// the transport boundary (`bc_`), the index domain (`dom`), the periodicity (`per_` / `periodic_`),
/// the SHARED aux channel (`aux` / `aux_ncomp_`), and the embedded-boundary domain (`eb_domain_` /
/// `eb_set_` / `domain_mask_` / `ws_cache_block_` / `geometry_mode_`). It names one subsystem: "where
/// the System lives and how it is laid out".
///
/// CONSTRUCTED FIRST: the members have an init-order dependency (`dm` sizes from `ba`, `aux` allocates
/// on `ba`/`dm`), so this struct owns the exact historical init-list. `System::Impl` holds a
/// `SystemDomain domain_` constructed BEFORE `fields_` / `stepper_`, whose back-pointers read the
/// layout through Impl's reference aliases.
///
/// STEPPER / FIELD VISIBILITY: `geom`, `pgeom_`, `polar_`, `aux`, `aux_ncomp_`, `ba`, `dm`, `bc_`,
/// `dom`, `per_`, `periodic_`, `cfg`, `geometry_mode_` and `eb_set_` are read by SystemStepper /
/// SystemFieldSolver / native_loader via `owner_->` / `P->`. Impl re-exposes EVERY member under its
/// exact historical name via a REFERENCE ALIAS (the proven `sp = blocks_.blocks` idiom), so the three
/// dependent headers and the MockImpl stay byte-unchanged -- the block closures that capture `&aux` by
/// address see a stable `&aux == &domain_.aux`.
///
/// OWNERSHIP CONTRACT: the layout (geom / ba / dm / bc_ / dom / per_ / aux width) is FROZEN AT BIND
/// (built at construction; widened only by the guarded structural setters, e.g. ensure_aux_width from
/// add_block, or set_disc_domain). The aux DATA is MUTABLE DURING RUN (re-derived by the field solve
/// each step). Not checkpointed here (the aux is re-derived; block state + ProgramRuntimeState carry
/// the restartable state).

namespace pops {
namespace runtime {
namespace system {

/// Data-only geometry/layout registry with the exact historical construction order.
struct SystemDomain {
  SystemConfig cfg;
  Geometry geom;
  // POLAR GEOMETRY (diocotron polar-grid project, Phase 2b). polar == true when cfg.geometry ==
  // "polar": the System runs on a global ring (r, theta) with the polar transport + polar Poisson.
  // pgeom is INERT in Cartesian -> bit-identical. dom/ba/dm always cover the INDEX space (nx x ny),
  // common to both geometries: only the indices -> physical space mapping (geom vs pgeom) changes.
  bool polar_;
  PolarGeometry pgeom_;
  BoxArray ba;
  DistributionMapping dm;
  BCRec
      bc_;  // transport BC (periodic or Foextrap per cfg.periodic; polar: physical r, periodic theta)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  int aux_ncomp_ = kAuxBaseComps;  // width of the SHARED aux channel (max over blocks; >= 3)

  // EMBEDDED-BOUNDARY / LEVEL-SET DOMAIN (T2 + T5-PR3, inert by default). eb_set_ == false: the mask
  // is "all active" and the transport path stays BIT-IDENTICAL. set_disc_domain fills eb_domain_ and
  // materializes domain_mask_ (0/1 cell-centered, 1 ghost). domain_mask_ / eb_domain_ are
  // STABLE-address members: the block closures read them by pointer at each step.
  detail::DiscDomain eb_domain_;
  bool eb_set_ = false;
  MultiFab
      domain_mask_;  // 0/1 cell-centered, same layout as the blocks (ba/dm), 1 ghost; empty while !eb_set_
  // At least one block requested wave_speed_cache (ADC-199, opt-in HLL cache): locks the switch to an
  // embedded-boundary transport mode (explicit rejection rather than a silently ignored cache).
  bool ws_cache_block_ = false;
  // TRANSPORT GEOMETRY MODE (T5-PR3): None (default) -> full Cartesian transport (bit-identical);
  // Staircase / CutCell -> the stepper routes to the masked / cut-cell advance. Effective only if a
  // domain is fixed (eb_set_) AND the block carries the matching embedded-boundary advance.
  GeometryMode geometry_mode_ = GeometryMode::None;

  // Number of radial / azimuthal cells in POLAR (0 => fall back to cfg.n).
  static int polar_nr(const SystemConfig& c) { return c.nr > 0 ? c.nr : c.n; }
  static int polar_ntheta(const SystemConfig& c) { return c.ntheta > 0 ? c.ntheta : c.n; }
  // INDEX domain: n x n square in Cartesian; nr x ntheta in polar (i = r, j = theta).
  static Box2D index_domain(const SystemConfig& c) {
    if (c.geometry == "polar")
      return Box2D::from_extents(polar_nr(c), polar_ntheta(c));
    return Box2D::from_extents(c.n, c.n);
  }
  // BoxArray of the INDEX domain. Cartesian (and polar mono-box, theta_boxes <= 1): ONE box covering
  // the whole domain -> STRICTLY bit-identical to the historical. Polar with theta_boxes > 1: theta
  // BANDS, each covering the whole radius [0, nr-1] and one contiguous azimuthal band. The bands tile
  // [0, ntheta-1] EXACTLY; check_geometry already validated the divisibility.
  static BoxArray index_boxarray(const SystemConfig& c) {
    if (c.geometry != "polar" || c.theta_boxes <= 1)
      return BoxArray(std::vector<Box2D>{index_domain(c)});
    const int nr = polar_nr(c), nth = polar_ntheta(c), nseg = c.theta_boxes;
    std::vector<Box2D> boxes;
    boxes.reserve(static_cast<std::size_t>(nseg));
    int base = nth / nseg, rem = nth % nseg, cur = 0;
    for (int k = 0; k < nseg; ++k) {
      const int len = base + (k < rem ? 1 : 0);
      boxes.push_back(Box2D{{0, cur}, {nr - 1, cur + len - 1}});
      cur += len;
    }
    return BoxArray(std::move(boxes));
  }

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodic by default
    if (c.geometry == "polar") {
      // POLAR: r (dir 0, xlo/xhi) carries a PHYSICAL BC (wall / free outflow, Foextrap); theta
      // (dir 1, ylo/yhi) is PERIODIC (the ring covers [0, 2pi)).
      b.xlo = b.xhi = BCType::Foextrap;
      b.ylo = b.yhi = BCType::Periodic;
      return b;
    }
    if (!c.periodic)
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  /// The exact historical System::Impl init-list, verbatim in order: cfg, geom, polar_, pgeom_, ba,
  /// dm (sizes from ba), bc_, dom, per_, periodic_, aux (allocates on ba/dm). The remaining members
  /// (eb_* / domain_mask_ / ws_cache_block_ / geometry_mode_) default-construct exactly as before.
  explicit SystemDomain(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        polar_(c.geometry == "polar"),
        pgeom_{index_domain(c), Real(c.r_min), Real(c.r_max)},
        ba(index_boxarray(c)),
        dm(ba.size(), n_ranks()),
        bc_(make_bc(c)),
        dom(index_domain(c)),
        per_{!polar_ && c.periodic, !polar_ && c.periodic},
        periodic_(!polar_ && c.periodic),
        aux(ba, dm, kAuxBaseComps, 1) {}

  /// Structured report (ADC-578 acceptance): the layout facts a runtime report enumerates.
  struct LayoutReport {
    bool polar;
    int nx, ny;
    int n_boxes;
    int aux_ncomp;
    bool periodic;
    bool eb_active;
  };
  LayoutReport layout_report() const {
    return LayoutReport{polar_,     dom.nx(),  dom.ny(), static_cast<int>(ba.size()),
                        aux_ncomp_, periodic_, eb_set_};
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
