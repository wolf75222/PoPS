/// @file
/// @brief Exact rank-local accepted moving-interval checkpoint envelope.
#pragma once

#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/moving_interval_geometry.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <limits>
#include <numeric>
#include <sstream>

namespace pops::runtime::program {
namespace moving_checkpoint_detail {
using checkpoint_detail::Reader;
using checkpoint_detail::Writer;
inline constexpr std::array<std::uint8_t, 8> magic{'P','O','P','S','E','X','0','3'};
inline void require(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(std::string("moving interval checkpoint: ")+reason);
}
template<int Dim> std::vector<std::uint8_t> topology(const MultiFab<Dim>& field, bool local=true) {
  Writer out;
  out.i32(Dim); out.i32(field.ncomp()); out.size(field.layout().size());
  out.i32(field.distribution().replicated());
  for (int axis=0; axis<Dim; ++axis) {
    out.i32(field.rank_space().origin()[axis]); out.i32(field.rank_space().extent()[axis]);
    if (local) out.i32(field.local_rank()[axis]);
  }
  for (std::size_t global=0; global<field.layout().size(); ++global) {
    for (int axis=0; axis<Dim; ++axis) {
      out.i32(field.layout()[global].lo[axis]); out.i32(field.layout()[global].hi[axis]);
      if (!field.distribution().replicated()) out.i32(field.distribution().owner(global)[axis]);
    }
  }
  if (local) {
    out.size(field.local_size());
    for (std::size_t patch=0; patch<field.local_size(); ++patch) out.size(field.global_index(patch));
  }
  return std::move(out).take();
}
template<int Dim> void write_fab(Writer& out, const Fab<Dim>& fab) {
  auto host=fab.create_host_mirror(); fab.copy_to_host(host);
  const auto cells=runtime::system::marshaling::checked_cell_count(fab.box());
  out.size(cells); out.i32(fab.ncomp());
  runtime::system::marshaling::for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
    for (int component=0; component<fab.ncomp(); ++component) {
      const Real value=host(runtime::system::marshaling::storage_ordinal(fab,index,component));
      require(std::isfinite(value), "nonfinite valid field value"); out.real(value);
    }
  });
}
template<int Dim> void read_fab(Reader& in, Fab<Dim>& fab) {
  const auto cells=runtime::system::marshaling::checked_cell_count(fab.box());
  require(in.size(8)==cells && in.i32()==fab.ncomp(), "field shape differs from installed authority");
  fab.set_val(0); auto host=fab.create_host_mirror(); fab.copy_to_host(host);
  runtime::system::marshaling::for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
    for (int component=0; component<fab.ncomp(); ++component) {
      const double value=in.real(); require(std::isfinite(value), "nonfinite valid field value");
      host(runtime::system::marshaling::storage_ordinal(fab,index,component))=Real(value);
    }
  });
  fab.copy_from_host(host);
}
template<int Dim> void write_field(Writer& out, const MultiFab<Dim>& field) {
  out.bytes(topology(field));
  for (std::size_t patch=0; patch<field.local_size(); ++patch) write_fab(out,field.fab(patch));
}
template<int Dim> MultiFab<Dim> read_field(Reader& in, const MultiFab<Dim>& prototype) {
  require(in.bytes()==topology(prototype), "field topology/component/rank authority differs");
  MultiFab<Dim> candidate=prototype;
  for (std::size_t patch=0; patch<candidate.local_size(); ++patch) read_fab(in,candidate.fab(patch));
  return candidate;
}
template<int Dim> void write_faces(Writer& out, const std::vector<nd::FaceField<Dim>>& faces,
                                 const MultiFab<Dim>& prototype, int components) {
  require(faces.size()==prototype.local_size(), "face patch count differs"); out.size(faces.size());
  for (std::size_t patch=0; patch<faces.size(); ++patch) {
    require(faces[patch].cell_box()==prototype.box(patch) && faces[patch].ncomp()==components,
            "face topology/components differ");
    write_fab(out,faces[patch].template field<0>());
  }
}
template<int Dim> std::vector<nd::FaceField<Dim>> read_faces(Reader& in,
    const MultiFab<Dim>& prototype, int components) {
  require(in.size(16)==prototype.local_size(), "face patch count differs");
  std::vector<nd::FaceField<Dim>> candidate;
  candidate.reserve(prototype.local_size());
  for (std::size_t patch=0; patch<prototype.local_size(); ++patch) {
    candidate.emplace_back(prototype.box(patch),components);
    read_fab(in,candidate.back().template field<0>());
  }
  return candidate;
}
inline void write_point(Writer& out, const multiblock::BoundaryEvaluationPoint& point) {
  out.string(point.clock); out.i64(point.tick); out.i32(point.level); out.i32(point.substep);
  out.i32(point.stage); out.i64(point.stage_fraction.numerator); out.i64(point.stage_fraction.denominator);
  out.real(point.dt); out.real(point.physical_time); out.string(point.graph_identity);
  out.string(point.rate_identity); out.string(point.application_identity);
}
inline multiblock::BoundaryEvaluationPoint read_point(Reader& in) {
  multiblock::BoundaryEvaluationPoint point;
  point.clock=in.string(); point.tick=in.i64(); point.level=in.i32(); point.substep=in.i32();
  point.stage=in.i32(); const auto numerator=in.i64(); const auto denominator=in.i64();
  point.stage_fraction=::pops::amr::Rational(numerator,denominator);
  point.dt=in.real(); point.physical_time=in.real(); point.graph_identity=in.string();
  point.rate_identity=in.string(); point.application_identity=in.string();
  require(!point.clock.empty() && point.stage==0 && point.stage_fraction==::pops::amr::Rational(0,1) &&
          point.level==0 && point.substep==0 && std::isfinite(point.dt) && point.dt>0 &&
          std::isfinite(point.physical_time), "receipt lacks complete interval time authority");
  return point;
}

/// Recompute geometry and Reynolds balance from independently stored accepted fields.
/// A wire image cannot declare itself admissible through its own checksum or residual.
template<int Dim> void validate_local(const MovingIntervalGeometry<Dim>& geometry,
                                     const MultiFab<Dim>& accepted_state, const std::string& identity,
                                     const AcceptedExchangeLedger& ledger) {
  if constexpr (Dim!=1) throw std::logic_error("moving checkpoint has no higher-dimensional provider");
  require(geometry.generation==0 ? (!geometry.last_receipt && geometry.last_interval.empty()) :
          (geometry.last_receipt.has_value() && !geometry.last_interval.empty()), "generation/receipt mismatch");
  require(!geometry.clock_authority.empty() && geometry.geometry_tolerance_authority.has_value() &&
          std::isfinite(*geometry.geometry_tolerance_authority) && *geometry.geometry_tolerance_authority>=0,
          "checkpoint requires explicitly installed clock and geometry tolerance authority");
  mf_arith_detail::require_same_layout(accepted_state,geometry.measures,"moving checkpoint measures",false);
  require(geometry.measures.ncomp()==1,"measure components must equal one");
  require(geometry.coordinates.size()==accepted_state.local_size() &&
          geometry.swept_volumes.size()==accepted_state.local_size(), "geometry patch count differs");
  std::map<std::tuple<std::string,std::string,std::string,std::string>,const ExchangeRecord*> records;
  for (const auto& record:ledger.records()) records.emplace(record.key(),&record);
  const auto require_record=[&](ExchangeRecord expected) {
    expected.qualify_runtime_point(geometry.last_receipt->point);
    const auto found=records.find(expected.key());
    require(found!=records.end(),"receipt lacks its exact accepted exchange occurrence");
    const auto& actual=*found->second;
    const bool matches=actual.orientation==expected.orientation && actual.face_measure==expected.face_measure &&
            actual.numerical_flux==expected.numerical_flux && actual.temporal_weight==expected.temporal_weight &&
            actual.multiplicity==expected.multiplicity && actual.trace_axis==expected.trace_axis &&
            actual.trace_side==expected.trace_side && actual.trace_component==expected.trace_component &&
            actual.exterior_trace==expected.exterior_trace &&
            actual.source_evaluation_identity==expected.source_evaluation_identity;
    if (!matches) {
      std::ostringstream details;
      details << "moving interval checkpoint: receipt quantities/support differ from accepted exchanges: "
              << expected.operation_identity << " " << expected.occurrence_identity << std::hexfloat
              << "; numerical_flux=" << actual.numerical_flux << " expected=" << expected.numerical_flux
              << "; face_measure=" << actual.face_measure << " expected=" << expected.face_measure
              << "; temporal_weight=" << actual.temporal_weight << " expected=" << expected.temporal_weight;
      throw std::invalid_argument(details.str());
    }
  };
  if (geometry.last_receipt) {
    const auto& receipt=*geometry.last_receipt;
    require(receipt.physical_frame==geometry.physical_frame && !receipt.quadrature_identity.empty() &&
            receipt.geometry_tolerance==*geometry.geometry_tolerance_authority &&
            receipt.point.clock==geometry.clock_authority && receipt.point.stage==0 &&
            receipt.point.stage_fraction==::pops::amr::Rational(0,1) && receipt.point.level==0 &&
            receipt.point.substep==0 && receipt.point.graph_identity.empty() &&
            receipt.point.rate_identity.empty() && receipt.point.application_identity.empty(),
            "receipt frame/quadrature/tolerance differs");
    ExchangeRecord point{"moving-interval","interval",identity,"endpoint-swept@1",1,1,0,1,1};
    point.qualify_runtime_point(receipt.point);
    require(point.evaluation_context==geometry.last_interval,"receipt interval identity differs");
    mf_arith_detail::require_same_layout(accepted_state,receipt.previous_state,"moving checkpoint old state");
    mf_arith_detail::require_same_layout(accepted_state,receipt.integrated_source,"moving checkpoint source");
    mf_arith_detail::require_same_layout(geometry.measures,receipt.previous_measures,"moving checkpoint old volumes");
    require(receipt.previous_coordinates.size()==accepted_state.local_size() &&
            receipt.physical_flux.size()==accepted_state.local_size() &&
            receipt.face_density.size()==accepted_state.local_size(),"receipt face patch count differs");
  }
  for (std::size_t patch=0; patch<accepted_state.local_size(); ++patch) {
    const auto& box=accepted_state.box(patch);
    require(geometry.coordinates[patch].cell_box()==box && geometry.coordinates[patch].ncomp()==1 &&
            geometry.swept_volumes[patch].cell_box()==box && geometry.swept_volumes[patch].ncomp()==1,
            "geometry face shape differs");
    if (geometry.last_receipt) {
      const auto& receipt=*geometry.last_receipt;
      require(receipt.previous_coordinates[patch].cell_box()==box && receipt.previous_coordinates[patch].ncomp()==1 &&
          receipt.physical_flux[patch].cell_box()==box && receipt.physical_flux[patch].ncomp()==accepted_state.ncomp() &&
          receipt.face_density[patch].cell_box()==box && receipt.face_density[patch].ncomp()==accepted_state.ncomp(),
          "receipt face shape differs");
    }
    auto volume=geometry.measures.fab(patch).create_host_mirror(); geometry.measures.fab(patch).copy_to_host(volume);
    auto position=geometry.coordinates[patch].template field<0>().create_host_mirror();
    geometry.coordinates[patch].template field<0>().copy_to_host(position);
    auto sweep=geometry.swept_volumes[patch].template field<0>().create_host_mirror();
    geometry.swept_volumes[patch].template field<0>().copy_to_host(sweep);
    const auto cells=static_cast<std::size_t>(accepted_state.box(patch).length(0));
    for (std::size_t cell=0; cell<cells; ++cell)
      require(volume(cell)>0 && std::isfinite(volume(cell)) && position(cell+1)>position(cell),
              "nonpositive accepted measure/geometry");
    if (!geometry.last_receipt) {
      for (std::size_t cell=0; cell<cells; ++cell)
        require(std::abs(volume(cell)-(position(cell+1)-position(cell)))<=
                Real(64)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),volume(cell)),
                "initial measures differ from accepted endpoints");
      for (std::size_t face=0; face<=cells; ++face)
        require(sweep(face)==0,"initial geometry has a swept volume without a receipt");
      continue;
    }
    const auto& receipt=*geometry.last_receipt;
    auto old_volume=receipt.previous_measures.fab(patch).create_host_mirror();
    receipt.previous_measures.fab(patch).copy_to_host(old_volume);
    auto old_position=receipt.previous_coordinates[patch].template field<0>().create_host_mirror();
    receipt.previous_coordinates[patch].template field<0>().copy_to_host(old_position);
    auto current=accepted_state.fab(patch).create_host_mirror(); accepted_state.fab(patch).copy_to_host(current);
    auto previous=receipt.previous_state.fab(patch).create_host_mirror(); receipt.previous_state.fab(patch).copy_to_host(previous);
    auto source=receipt.integrated_source.fab(patch).create_host_mirror(); receipt.integrated_source.fab(patch).copy_to_host(source);
    auto flux=receipt.physical_flux[patch].template field<0>().create_host_mirror();
    receipt.physical_flux[patch].template field<0>().copy_to_host(flux);
    auto density=receipt.face_density[patch].template field<0>().create_host_mirror();
    receipt.face_density[patch].template field<0>().copy_to_host(density);
    const Real tolerance=receipt.geometry_tolerance;
    for (std::size_t face=0; face<=cells; ++face)
      require(std::abs((position(face)-old_position(face))-sweep(face))<=tolerance,
              "receipt sweep differs from endpoint displacement");
    for (std::size_t cell=0; cell<cells; ++cell) {
      require(old_volume(cell)>0 && old_position(cell+1)>old_position(cell) &&
          volume(cell)==position(cell+1)-position(cell) &&
          old_volume(cell)==old_position(cell+1)-old_position(cell) &&
          std::abs((volume(cell)-old_volume(cell))-(sweep(cell+1)-sweep(cell)))<=tolerance,
          "receipt geometric conservation law failed");
      Index<Dim> index=accepted_state.box(patch).lo; index[0]+=static_cast<int>(cell);
      for (int component=0; component<accepted_state.ncomp(); ++component) {
        const Real q0=previous(runtime::system::marshaling::storage_ordinal(receipt.previous_state.fab(patch),index,component))*old_volume(cell);
        const Real q1=current(runtime::system::marshaling::storage_ordinal(accepted_state.fab(patch),index,component))*volume(cell);
        const Real amount=source(runtime::system::marshaling::storage_ordinal(receipt.integrated_source.fab(patch),index,component));
        const auto offset=cell+(cells+1)*static_cast<std::size_t>(component);
        const Real left=flux(offset)-density(offset)*sweep(cell);
        const Real right=flux(offset+1)-density(offset+1)*sweep(cell+1);
        const Real scale=std::max({Real(1),std::abs(q0),std::abs(q1),std::abs(amount),std::abs(left),std::abs(right)});
        const Real residual=(q1-q0)+(right-left)-amount;
        require(std::isfinite(q0) && std::isfinite(q1) && std::isfinite(left) && std::isfinite(right) &&
                std::isfinite(amount) && std::isfinite(scale) && std::isfinite(residual),
                "nonfinite derived receipt quantity/residual");
        require(std::abs(residual)<=Real(64)*std::numeric_limits<Real>::epsilon()*scale,
                "receipt independent Reynolds balance failed");
        require_record(ExchangeRecord{"source:"+identity+"/component:"+std::to_string(component),
            "cell:"+std::to_string(index[0]),identity,receipt.quadrature_identity,1,1.,double(amount),1.,1});
        for (int side=0; side<2; ++side) {
          const auto occurrence="cell:"+std::to_string(index[0])+"/side:"+std::to_string(side);
          const int outward=side==0 ? -1 : 1;
          require_record(ExchangeRecord{"amount:"+identity+"/component:"+std::to_string(component),
              occurrence,identity,receipt.quadrature_identity,outward,1.,double(side==0 ? left : right),1.,1});
          if (component==0)
            require_record(ExchangeRecord{"geometry:"+identity,occurrence,identity,receipt.quadrature_identity,
                outward,1.,double(sweep(cell+side)),1.,1});
        }
      }
    }
  }
}
} // namespace moving_checkpoint_detail

template<int Dim> struct MovingCheckpointCandidate {
  AcceptedExchangeLedger exchanges;
  std::map<std::string,MovingIntervalGeometry<Dim>> geometry;
  std::map<int,MultiFab<Dim>> states;
};

template<int Dim> void require_moving_checkpoint_lifecycle(const MovingCheckpointCandidate<Dim>& candidate,
                                                          double accepted_time, int macro_step) {
  moving_checkpoint_detail::require(std::isfinite(accepted_time) && macro_step>=0,
                                    "invalid enclosing accepted lifecycle authority");
  for (const auto& [identity,geometry]:candidate.geometry) {
    if (!geometry.last_receipt) continue;
    const auto& point=geometry.last_receipt->point;
    moving_checkpoint_detail::require(std::isfinite(point.physical_time+point.dt) &&
        point.physical_time+point.dt==accepted_time && point.tick>=0 &&
        point.tick==static_cast<std::int64_t>(macro_step)-1,
        "receipt interval end/tick differs from enclosing accepted lifecycle");
  }
}

template<int Dim, class StateLookup> std::vector<std::uint8_t> checkpoint_moving_intervals(
    const AcceptedExchangeLedger& ledger, const std::map<std::string,MovingIntervalGeometry<Dim>>& geometries,
    StateLookup state) {
  using namespace moving_checkpoint_detail;
  Writer out; out.raw(magic); out.bytes(ledger.checkpoint(true)); out.size(geometries.size());
  for (const auto& [identity,geometry]:geometries) {
    const auto& physical=state(geometry.runtime_block); validate_local(geometry,physical,identity,ledger);
    out.string(identity); out.i32(geometry.runtime_block); out.string(geometry.physical_frame);
    out.string(geometry.clock_authority); out.real(*geometry.geometry_tolerance_authority);
    out.u64(geometry.generation); out.string(geometry.last_interval); write_field(out,physical);
    write_field(out,geometry.measures); write_faces(out,geometry.coordinates,physical,1);
    write_faces(out,geometry.swept_volumes,physical,1); out.u64(geometry.last_receipt ? 1 : 0);
    if (geometry.last_receipt) {
      const auto& receipt=*geometry.last_receipt;
      write_point(out,receipt.point); out.string(receipt.physical_frame); out.string(receipt.quadrature_identity);
      out.real(receipt.geometry_tolerance); write_field(out,receipt.previous_state);
      write_field(out,receipt.previous_measures); write_field(out,receipt.integrated_source);
      write_faces(out,receipt.previous_coordinates,physical,1);
      write_faces(out,receipt.physical_flux,physical,physical.ncomp());
      write_faces(out,receipt.face_density,physical,physical.ncomp());
    }
  }
  return std::move(out).take();
}

template<int Dim, class StateLookup> MovingCheckpointCandidate<Dim> read_moving_checkpoint(
    std::span<const std::uint8_t> bytes, const AcceptedExchangeLedger& ledger,
    const std::map<std::string,MovingIntervalGeometry<Dim>>& declarations, StateLookup state) {
  using namespace moving_checkpoint_detail;
  MovingCheckpointCandidate<Dim> result;
  if (declarations.empty()) {
    result.exchanges=AcceptedExchangeLedger::from_checkpoint(bytes);
  } else {
    Reader in(bytes); in.expect_raw(magic); result.exchanges=AcceptedExchangeLedger::from_checkpoint(in.bytes());
    require(in.size(32)==declarations.size(), "geometry declaration count differs");
    for (const auto& [identity,declaration]:declarations) {
      require(in.string()==identity && in.i32()==declaration.runtime_block &&
              in.string()==declaration.physical_frame, "geometry identity/block/frame differs");
      require(declaration.geometry_tolerance_authority.has_value(),"installed geometry lacks tolerance authority");
      require(in.string()==declaration.clock_authority && in.real()==*declaration.geometry_tolerance_authority,
              "geometry clock/tolerance authority differs from installed declaration");
      MovingIntervalGeometry<Dim> geometry=declaration;
      geometry.generation=in.u64(); geometry.last_interval=in.string();
      const auto& physical=state(declaration.runtime_block);
      auto accepted=read_field(in,physical); geometry.measures=read_field(in,declaration.measures);
      geometry.coordinates=read_faces(in,physical,1); geometry.swept_volumes=read_faces(in,physical,1);
      int first=std::numeric_limits<int>::max(), last=std::numeric_limits<int>::min();
      for (std::size_t global=0; global<physical.layout().size(); ++global) {
        const auto& box=physical.layout()[global]; first=std::min(first,box.lo[0]); last=std::max(last,box.hi[0]);
      }
      for (std::size_t patch=0; patch<physical.local_size(); ++patch) {
        if (physical.box(patch).lo[0]!=first && physical.box(patch).hi[0]!=last) continue;
        auto installed=declaration.coordinates[patch].template field<0>().create_host_mirror();
        declaration.coordinates[patch].template field<0>().copy_to_host(installed);
        auto restored=geometry.coordinates[patch].template field<0>().create_host_mirror();
        geometry.coordinates[patch].template field<0>().copy_to_host(restored);
        if (physical.box(patch).lo[0]==first)
          require(installed(0)==restored(0),"fixed lower physical boundary differs from installed geometry");
        const auto right=static_cast<std::size_t>(physical.box(patch).length(0));
        if (physical.box(patch).hi[0]==last)
          require(installed(right)==restored(right),"fixed upper physical boundary differs from installed geometry");
      }
      const auto receipt_present=in.u64(); require(receipt_present<=1, "invalid receipt tag");
      geometry.last_receipt.reset();
      if (receipt_present) {
        auto& receipt=geometry.last_receipt.emplace(); receipt.point=read_point(in);
        receipt.physical_frame=in.string(); receipt.quadrature_identity=in.string(); receipt.geometry_tolerance=Real(in.real());
        receipt.previous_state=read_field(in,physical); receipt.previous_measures=read_field(in,declaration.measures);
        receipt.integrated_source=read_field(in,physical); receipt.previous_coordinates=read_faces(in,physical,1);
        receipt.physical_flux=read_faces(in,physical,physical.ncomp()); receipt.face_density=read_faces(in,physical,physical.ncomp());
      }
      validate_local(geometry,accepted,identity,result.exchanges);
      require(result.states.emplace(declaration.runtime_block,std::move(accepted)).second,
              "duplicate geometry owner");
      result.geometry.emplace(identity,std::move(geometry));
    }
    in.finish();
  }
  require(result.exchanges.same_integral_declarations(ledger), "integral-state declarations differ");
  return result;
}

/// Authenticate global metadata before deriving any rank-dependent endpoint schedule.
/// All local allocation/host-copy work converges before the next collective.
template<int Dim> void require_moving_checkpoint_agrees_collectively(
    const MovingCheckpointCandidate<Dim>& candidate, const ExecutionLane& lane) {
  using namespace moving_checkpoint_detail;
  std::vector<std::uint8_t> metadata;
  std::exception_ptr error;
  try {
    Writer out; out.size(candidate.geometry.size());
    for (const auto& [identity,geometry]:candidate.geometry) {
      out.string(identity); out.i32(geometry.runtime_block); out.string(geometry.physical_frame);
      out.string(geometry.clock_authority); out.real(*geometry.geometry_tolerance_authority);
      out.u64(geometry.generation); out.string(geometry.last_interval);
      out.bytes(topology(candidate.states.at(geometry.runtime_block),false));
      out.u64(geometry.last_receipt ? 1 : 0);
      if (geometry.last_receipt) {
        write_point(out,geometry.last_receipt->point); out.string(geometry.last_receipt->quadrature_identity);
        out.real(geometry.last_receipt->geometry_tolerance);
      }
    }
    metadata=std::move(out).take();
  } catch (...) { error=std::current_exception(); }
  collectively_rethrow_exception(error,lane,"moving checkpoint metadata preparation failed collectively");
  const std::string_view metadata_view(reinterpret_cast<const char*>(metadata.data()),metadata.size());
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"moving-checkpoint",metadata_view}},lane))
    throw std::invalid_argument("moving checkpoint global time/topology/declaration differs across ranks");
  for (const auto& [identity,geometry]:candidate.geometry) {
    const auto& physical=candidate.states.at(geometry.runtime_block);
    std::vector<std::size_t> ordered;
    std::vector<std::vector<Real>> endpoints, local_endpoints;
    try {
      const std::size_t width=3+2*static_cast<std::size_t>(physical.ncomp());
      ordered.resize(physical.layout().size());
      std::iota(ordered.begin(),ordered.end(),0);
      std::sort(ordered.begin(),ordered.end(),[&](auto a,auto b) { return physical.layout()[a].lo[0]<physical.layout()[b].lo[0]; });
      require(!ordered.empty(),"empty global topology");
      for (std::size_t row=1; row<ordered.size(); ++row)
        require(physical.layout()[ordered[row-1]].hi[0]+1==physical.layout()[ordered[row]].lo[0],
                "global intervals do not tile contiguously");
      endpoints.assign(ordered.size(),std::vector<Real>(2*width,0));
      local_endpoints.assign(physical.local_size(),std::vector<Real>(2*width,0));
      for (std::size_t patch=0; patch<physical.local_size(); ++patch) {
        const auto cells=static_cast<std::size_t>(physical.box(patch).length(0));
        auto position=geometry.coordinates[patch].template field<0>().create_host_mirror();
        geometry.coordinates[patch].template field<0>().copy_to_host(position);
        auto sweep=geometry.swept_volumes[patch].template field<0>().create_host_mirror();
        geometry.swept_volumes[patch].template field<0>().copy_to_host(sweep);
        for (std::size_t side=0; side<2; ++side) {
          const auto face=side*cells, base=side*width;
          local_endpoints[patch][base]=position(face);
          local_endpoints[patch][base+1]=position(face);
          local_endpoints[patch][base+2]=sweep(face);
        }
        if (geometry.last_receipt) {
          const auto& receipt=*geometry.last_receipt;
          auto old=receipt.previous_coordinates[patch].template field<0>().create_host_mirror();
          receipt.previous_coordinates[patch].template field<0>().copy_to_host(old);
          auto flux=receipt.physical_flux[patch].template field<0>().create_host_mirror();
          receipt.physical_flux[patch].template field<0>().copy_to_host(flux);
          auto density=receipt.face_density[patch].template field<0>().create_host_mirror();
          receipt.face_density[patch].template field<0>().copy_to_host(density);
          for (std::size_t side=0; side<2; ++side) {
            const auto face=side*cells, base=side*width;
            local_endpoints[patch][base]=old(face);
            for (int component=0; component<physical.ncomp(); ++component) {
              const auto offset=face+(cells+1)*static_cast<std::size_t>(component);
              local_endpoints[patch][base+3+2*component]=flux(offset);
              local_endpoints[patch][base+4+2*component]=density(offset);
            }
          }
        }
      }
    } catch (...) { error=std::current_exception(); }
    collectively_rethrow_exception(error,lane,"moving checkpoint endpoint preparation failed collectively");
    bool invalid=false;
    for (std::size_t row=0; row<ordered.size(); ++row) {
      const auto global=ordered[row], local=physical.local_index_of(global);
      const int owner=physical.distribution().replicated() ? 0 : static_cast<int>(
          physical.rank_space().linear_rank(physical.distribution().owner(global)));
      if (local!=MultiFab<Dim>::not_local)
        std::copy(local_endpoints[local].begin(),local_endpoints[local].end(),endpoints[row].begin());
      broadcast_bytes_inplace(reinterpret_cast<char*>(endpoints[row].data()),endpoints[row].size()*sizeof(Real),lane,owner);
      if (local!=MultiFab<Dim>::not_local && endpoints[row]!=local_endpoints[local]) invalid=true;
      const auto width=endpoints[row].size()/2;
      if (row && !std::equal(endpoints[row].begin(),endpoints[row].begin()+width,endpoints[row-1].begin()+width)) invalid=true;
    }
    const auto width=endpoints.front().size()/2;
    if (endpoints.front()[0]!=endpoints.front()[1] || endpoints.back()[width]!=endpoints.back()[width+1]) invalid=true;
    if (all_reduce_max(invalid ? 1L : 0L,lane))
      throw std::invalid_argument("moving checkpoint shared-face/fixed-boundary values disagree");
  }
}
} // namespace pops::runtime::program
