/// @file
/// @brief Prepared conservative face diffusion; no transport or solve authority.
#pragma once
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::runtime::program {
enum class DiffusiveBoundaryKind { periodic, value, conormal };
template<int Dim> struct DiffusiveBoundary {
  DiffusiveBoundaryKind kind = DiffusiveBoundaryKind::periodic;
  Real value = Real(0);
  std::array<Real,Dim> slope{};
  POPS_HD Real trace(const Geometry<Dim>& geometry, const Index<Dim>& face, int axis) const {
    Real result = value;
    for (int d=0; d<Dim; ++d)
      result += slope[d] * (d == axis ? geometry.face_coordinate(d,face[d])
                                     : geometry.cell_coordinate(d,face[d]));
    return result;
  }
};

/// One preparation per Program evaluation/implicit operator, reused for every trial.
/// The law factory supplies W(U), positive diagonal A(U,fields), and dW/dU at each cell.
/// Neighbor differences are taken AFTER evaluating W: no discrete chain rule is substituted.
template<int Dim> class PreparedDiffusion {
  static_assert(Dim == 1 || Dim == 2, "selected diffusion matrix is Cartesian Dim1/Dim2");
  using Field = MultiFab<Dim>;
  using Boundary = PreparedScalarBoundarySession<Dim>;
  Geometry<Dim> geometry_;
  std::array<DiffusiveBoundary<Dim>,2*Dim> physical_;
  Field variable_, coefficients_, status_;
  std::shared_ptr<Boundary> state_boundary_, variable_boundary_, coefficient_boundary_;
  std::vector<nd::FaceField<Dim>> faces_;
  Real frequency_ = Real(0);
  bool evaluated_ = false;
 public:
  template<class Context>
  PreparedDiffusion(Context& ctx, Field& prototype,
                    std::array<DiffusiveBoundary<Dim>,2*Dim> physical)
      : geometry_(ctx.geometry()), physical_(physical),
        variable_(prototype.layout(),prototype.distribution(),prototype.local_rank(),2,prototype.ghosts()),
        coefficients_(prototype.layout(),prototype.distribution(),prototype.local_rank(),Dim,prototype.ghosts()),
        status_(prototype.layout(),prototype.distribution(),prototype.local_rank(),1,prototype.ghosts()) {
    if (ctx.prepared_execution_lane().size() != 1)
      throw std::invalid_argument("diffusion currently qualifies one MPI rank only");
    if (prototype.ncomp() != 1)
      throw std::invalid_argument("diffusion requires one scalar evolved component");
    state_boundary_ = ctx.prepare_mesh_boundary_session(prototype,ctx.prepared_execution_lane());
    variable_boundary_ = ctx.prepare_mesh_boundary_session(variable_,ctx.prepared_execution_lane());
    coefficient_boundary_ = ctx.prepare_mesh_boundary_session(coefficients_,ctx.prepared_execution_lane());
    for (int axis=0; axis<Dim; ++axis) {
      if (geometry_.domain().length(axis)<2 || prototype.ghosts()[axis]<1)
        throw std::invalid_argument("diffusion requires at least two cells and one halo per axis");
      for (int side=0;side<2;++side) {
        bool periodic = state_boundary_->topology().is_periodic(
            Face<Dim>{axis,side == 0 ? BoundarySide::lower : BoundarySide::upper});
        if (periodic != (physical_[2*axis+side].kind == DiffusiveBoundaryKind::periodic))
          throw std::invalid_argument("physical diffusive boundary differs from bound mesh topology");
      }
    }
    for(std::size_t local=0;local<prototype.local_size();++local)
      faces_.emplace_back(prototype.box(local),2);
  }

  template<class LawFactory>
  void apply(Field& input, Field& output, LawFactory factory) {
    evaluated_ = false;
    if (input.shares_storage_with(output) || input.ncomp()!=1 || output.ncomp()!=1 ||
        input.local_size()!=variable_.local_size() || output.local_size()!=variable_.local_size())
      throw std::invalid_argument("diffusion input/output do not match prepared scalar storage");
    for(std::size_t local=0;local<input.local_size();++local) {
      const auto law = factory(local);
      const auto w = variable_.fab(local).view();
      const auto a = coefficients_.fab(local).view();
      const auto status = status_.fab(local).view();
      for_each_cell(input.box(local),[=] POPS_HD(const Index<Dim>& cell) {
        const auto values = law(cell);
        bool valid = Kokkos::isfinite(values[0]) && Kokkos::isfinite(values[Dim+1]) && values[Dim+1]>=0;
        w(cell,0)=values[0]; w(cell,1)=values[Dim+1];
        for(int axis=0;axis<Dim;++axis) {
          a(cell,axis)=values[axis+1];
          valid = valid && Kokkos::isfinite(values[axis+1]) && values[axis+1]>0;
        }
        status(cell,0)=valid ? Real(0) : Real(1);
      });
    }
    device_fence();
    if(reduce_max_local(status_)!=Real(0))
      throw std::runtime_error("diffusive constitutive evaluation is non-finite or not positive");
    state_boundary_->fill_halo(input);
    variable_boundary_->fill_halo(variable_);
    coefficient_boundary_->fill_halo(coefficients_);
    const auto geometry=geometry_;
    const auto physical=physical_;
    for(std::size_t local=0;local<input.local_size();++local) {
      const auto q=std::as_const(input).fab(local).view();
      const auto w=std::as_const(variable_).fab(local).view();
      const auto a=std::as_const(coefficients_).fab(local).view();
      const auto faces=faces_[local].view();
      for(int axis=0;axis<Dim;++axis) {
        const auto face_values=faces.axes[axis];
        for_each_cell(nd::face_box(input.box(local),axis),[=] POPS_HD(const Index<Dim>& face) {
          Index<Dim> left=face, right=face;
          --left[axis];
          const bool lower=face[axis]==geometry.domain().lo[axis];
          const bool upper=face[axis]==geometry.domain().hi[axis]+1;
          const auto boundary=physical[2*axis+(upper ? 1 : 0)];
          const Real h=geometry.spacing(axis);
          Real flux=Real(0), conductance=Real(0);
          if((lower || upper) && boundary.kind!=DiffusiveBoundaryKind::periodic) {
            const Index<Dim> center=lower ? right : left;
            const Real orientation=lower ? Real(-1) : Real(1);
            if(boundary.kind==DiffusiveBoundaryKind::conormal)
              flux=orientation*boundary.trace(geometry,face,axis);
            else {
              Index<Dim> inside=center; inside[axis]+=lower ? 1 : -1;
              const Real coefficient=Real(1.5)*a(center,axis)-Real(0.5)*a(inside,axis);
              flux=orientation*coefficient*Real(2)*(boundary.trace(geometry,face,axis)-w(center,0))/h;
              conductance=Real(2)*coefficient*w(center,1)/h;
              if(!(coefficient>0)) flux=std::numeric_limits<Real>::quiet_NaN();
            }
          } else {
            const Real coefficient=Real(0.5)*(a(left,axis)+a(right,axis));
            flux=coefficient*(w(right,0)-w(left,0))/h;
            const Real difference=q(right,0)-q(left,0);
            const Real secant=difference != 0 ? (w(right,0)-w(left,0))/difference
                                              : Real(0.5)*(w(right,1)+w(left,1));
            conductance=coefficient*secant/h;
          }
          face_values(face,0)=flux;
          face_values(face,1)=conductance;
        });
      }
      const auto result=output.fab(local).view();
      const auto status=status_.fab(local).view();
      for_each_cell(output.box(local),[=] POPS_HD(const Index<Dim>& cell) {
        Real divergence=0,frequency=0;
        for(int axis=0;axis<Dim;++axis) {
          Index<Dim> upper=cell; ++upper[axis];
          divergence+=(faces.axes[axis](upper,0)-faces.axes[axis](cell,0))/geometry.spacing(axis);
          frequency+=(faces.axes[axis](upper,1)+faces.axes[axis](cell,1))/geometry.spacing(axis);
        }
        result(cell,0)=divergence;
        status(cell,0)=Kokkos::isfinite(divergence) && Kokkos::isfinite(frequency) && frequency>=0
                        ? frequency : std::numeric_limits<Real>::infinity();
      });
    }
    device_fence();
    frequency_=reduce_max_local(status_);
    if(!std::isfinite(frequency_)) throw std::runtime_error("diffusive face evaluation is invalid");
    evaluated_=true;
  }
  Real explicit_frequency() const {
    if(!evaluated_) throw std::logic_error("diffusive stability requires a completed face evaluation");
    return frequency_;
  }
  const auto& faces() const { return faces_; }
  const auto& geometry() const { return geometry_; }
};
} // namespace pops::runtime::program
