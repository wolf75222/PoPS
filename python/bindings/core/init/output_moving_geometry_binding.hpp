#pragma once

#include "output_geometry_binding.hpp"
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/moving_interval_checkpoint.hpp>
#include <pops/core/identity/prepared_provider.hpp>

namespace pops::python::detail {

/// Project real accepted endpoints and measures by exact native owners. This
/// global visualization image is not a physical reduction: replicated storage
/// has one authoritative broadcast root and is never summed per MPI rank.
template<int Dim>
py::dict moving_output_geometry_snapshot(const System<Dim>& system,
                                         const std::string& identity,
                                         const std::string& frame) {
  runtime::program::ProgramContext<Dim> context(const_cast<System<Dim>*>(&system));
  const auto& lane=context.prepared_execution_lane();
  std::vector<std::vector<Real>> patches;
  std::vector<Real> positions,volumes;
  std::exception_ptr error;
  const runtime::program::MovingIntervalGeometry<Dim>* geometry=nullptr;
  std::string declaration;
  {
    py::gil_scoped_release release;
    try {
      if constexpr(Dim!=1) throw std::logic_error("moving output needs a prepared 1D endpoint provider");
      geometry=&context.moving_interval_geometry(identity);
      if(geometry->physical_frame!=frame || geometry->measures.ncomp()!=1 ||
          geometry->coordinates.size()!=geometry->measures.local_size())
        throw std::invalid_argument("moving output geometry authority/shape differs");
      const auto& mapping=context.runtime_state().block_map_;
      const auto block=std::find(mapping.begin(),mapping.end(),geometry->runtime_block);
      if(block==mapping.end()) throw std::invalid_argument("moving output lacks its exact Program state route");
      runtime::program::moving_checkpoint_detail::validate_local(*geometry,
          context.state(static_cast<int>(block-mapping.begin())),identity,context.runtime_state().accepted_exchanges_);
      const auto reference=context.geometry();
      const auto& measures=geometry->measures;
      positions.resize(static_cast<std::size_t>(reference.domain().length(0))+1);
      volumes.resize(positions.size()-1);
      patches.resize(measures.box_array().size());
      ExactContractBuilder contract;
      contract.text("pops.moving-output.v1").text(identity).text(frame).scalar(geometry->generation)
          .scalar(std::int32_t(measures.distribution().replicated())).scalar(std::uint64_t(patches.size()));
      contract.text(geometry->clock_authority).scalar(*geometry->geometry_tolerance_authority)
          .text(geometry->last_interval).scalar(std::int32_t(geometry->last_receipt.has_value()));
      if(geometry->last_receipt) {
        runtime::program::moving_checkpoint_detail::Writer point;
        runtime::program::moving_checkpoint_detail::write_point(point,geometry->last_receipt->point);
        const auto bytes=std::move(point).take();
        contract.bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()),bytes.size()))
            .text(geometry->last_receipt->quadrature_identity);
      }
      contract.text(lane.identity()).scalar(std::int32_t(lane.size()));
      for(int axis=0;axis<Dim;++axis)
        contract.scalar(std::int32_t(measures.rank_space().origin()[axis]))
            .scalar(std::int32_t(measures.rank_space().extent()[axis]))
            .scalar(std::int32_t(reference.domain().lo[axis])).scalar(std::int32_t(reference.domain().hi[axis]))
            .scalar(reference.lower()[axis]).scalar(reference.upper()[axis]);
      std::vector<bool> represented(volumes.size(),false);
      if(measures.ghosts()[0]!=0) throw std::invalid_argument("moving output measures require exact valid-cell storage");
      for(std::size_t global=0;global<patches.size();++global) {
        const auto box=measures.box_array()[global];
        if(box.lo[0]<reference.domain().lo[0] || box.hi[0]>reference.domain().hi[0] || box.empty())
          throw std::invalid_argument("moving output patch lies outside its reference topology");
        for(int cell=box.lo[0];cell<=box.hi[0];++cell) {
          const auto offset=static_cast<std::size_t>(cell-reference.domain().lo[0]);
          if(represented[offset]) throw std::invalid_argument("moving output has overlapping physical cells");
          represented[offset]=true;
        }
        const int owner=measures.distribution().replicated()?0:static_cast<int>(
            measures.rank_space().linear_rank(measures.distribution().owner(global)));
        contract.scalar(std::int32_t(box.lo[0])).scalar(std::int32_t(box.hi[0])).scalar(std::int32_t(owner));
        const auto cells=static_cast<std::size_t>(box.length(0));
        patches[global].resize(2*cells+1);
        const auto local=measures.local_index_of(global);
        if(local==MultiFab<Dim>::not_local) continue;
        const auto& coordinate=geometry->coordinates[local].template field<0>();
        if(geometry->coordinates[local].cell_box()!=box || coordinate.ncomp()!=1)
          throw std::invalid_argument("moving output endpoint storage differs from exact topology");
        auto endpoints=coordinate.create_host_mirror(); coordinate.copy_to_host(endpoints);
        auto amounts=measures.fab(local).create_host_mirror(); measures.fab(local).copy_to_host(amounts);
        for(std::size_t face=0;face<=cells;++face) patches[global][face]=endpoints(face);
        for(std::size_t cell=0;cell<cells;++cell) patches[global][cells+1+cell]=amounts(cell);
      }
      if(!std::all_of(represented.begin(),represented.end(),[](bool value){return value;}))
        throw std::invalid_argument("moving output topology omits physical cells");
      declaration=std::move(contract).release();
    } catch(...) { error=std::current_exception(); }
    collectively_rethrow_exception(error,lane,"moving output preparation failed collectively");
    if(!all_ranks_agree_exact_ordered_byte_pairs({{"moving-output",declaration}},lane))
      throw std::invalid_argument("moving output topology/owners differ between ranks");
    const auto& measures=geometry->measures;
    const auto reference=context.geometry();
    std::fill(positions.begin(),positions.end(),std::numeric_limits<Real>::quiet_NaN());
    for(std::size_t global=0;global<patches.size();++global) {
      const auto box=measures.box_array()[global];
      const auto cells=static_cast<std::size_t>(box.length(0));
      const int owner=measures.distribution().replicated()?0:static_cast<int>(
          measures.rank_space().linear_rank(measures.distribution().owner(global)));
      broadcast_bytes_inplace(reinterpret_cast<char*>(patches[global].data()),
                               patches[global].size()*sizeof(Real),lane,owner);
      const auto begin=static_cast<std::size_t>(box.lo[0]-reference.domain().lo[0]);
      for(std::size_t face=0;face<=cells;++face) {
        if(!std::isnan(positions[begin+face]) && positions[begin+face]!=patches[global][face])
          throw std::invalid_argument("moving output shared endpoints differ");
        positions[begin+face]=patches[global][face];
      }
      for(std::size_t cell=0;cell<cells;++cell) volumes[begin+cell]=patches[global][cells+1+cell];
    }
    for(std::size_t cell=0;cell<volumes.size();++cell)
      if(!std::isfinite(positions[cell]) || !std::isfinite(positions[cell+1]) ||
          !std::isfinite(volumes[cell]) || !(volumes[cell]>0) ||
          volumes[cell]!=positions[cell+1]-positions[cell])
        throw std::invalid_argument("moving output measures differ from accepted physical endpoints");
  }
  py::array_t<double> nodes({static_cast<py::ssize_t>(positions.size()),py::ssize_t(1)});
  py::array_t<double> amounts(static_cast<py::ssize_t>(volumes.size()));
  std::copy(positions.begin(),positions.end(),nodes.mutable_data());
  std::copy(volumes.begin(),volumes.end(),amounts.mutable_data());
  nodes.attr("setflags")(false); amounts.attr("setflags")(false);
  py::dict result;
  result["generation"]=geometry->generation;
  result["node_coordinates"]=std::move(nodes); result["cell_volumes"]=std::move(amounts);
  return result;
}

} // namespace pops::python::detail
