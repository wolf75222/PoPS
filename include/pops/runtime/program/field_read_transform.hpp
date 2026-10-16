#pragma once

#include <pops/runtime/program/field_read_transport.hpp>

namespace pops::runtime::program {

struct ClosedFieldTransformReads {
  std::string inputs;
  std::string coordinates;
};

// The two journals are written by the same generated Kokkos equation kernel.
// Free reads exclude untaken branches; coordinates record the complete input
// value, including masked cells, without changing its stored floating bits.
// This immutable description still grants no authority and acquits no Field.
template <int Dim>
struct CapturedTransformReads {
  CapturedFieldReads<Dim> inputs, coordinates;

  static CapturedTransformReads prepare(const MultiFab<Dim>& source, std::uint64_t free_bindings,
                                        const ExecutionLane& lane, std::uint64_t capacity) {
    field_read_detail::Budget budget(capacity);
    field_read_detail::phase(
        lane,
        [&] {
          if (source.ncomp() < 1)
            throw std::invalid_argument("coordinate capture requires a typed Field");
          budget.charge(sizeof(CapturedTransformReads));
        },
        "transform capture preflight failed");
    auto inputs = CapturedFieldReads<Dim>::prepare(source, free_bindings, lane, budget.remaining());
    field_read_detail::phase(
        lane, [&] { budget.charge(inputs.retained_bytes); },
        "transform capture resource preflight failed");
    auto coordinates =
        CapturedFieldReads<Dim>::prepare(source, source.ncomp(), lane, budget.remaining());
    return {std::move(inputs), std::move(coordinates)};
  }

  ClosedFieldTransformReads close(const MultiFab<Dim>& source, std::string_view equation_schema,
                                  std::string_view coordinate_schema, std::string_view support,
                                  std::span<const std::string_view> authorities,
                                  const ExecutionLane& lane, std::uint64_t capacity) const {
    field_read_detail::Budget budget(capacity);
    field_read_detail::phase(
        lane,
        [&] {
          budget.charge(sizeof(CapturedTransformReads));
          budget.charge(sizeof(ClosedFieldTransformReads));
          budget.charge(coordinates.retained_bytes);
        },
        "transform closure resource preflight failed");
    ClosedFieldTransformReads result;
    result.inputs =
        inputs.close(source, equation_schema, support, authorities, lane, budget.remaining());
    field_read_detail::Budget coordinate_budget(capacity);
    std::vector<std::string_view> components;
    field_read_detail::phase(
        lane,
        [&] {
          coordinate_budget.charge(sizeof(CapturedTransformReads));
          coordinate_budget.charge(sizeof(ClosedFieldTransformReads));
          coordinate_budget.charge(inputs.retained_bytes);
          coordinate_budget.charge(result.inputs.size());
          coordinate_budget.charge(source.ncomp(), sizeof(std::string_view));
          components.assign(static_cast<std::size_t>(source.ncomp()), "typed-coordinate-component");
        },
        "coordinate closure resource preflight failed");
    result.coordinates = coordinates.close(source, coordinate_schema, support, components, lane,
                                           coordinate_budget.remaining());
    return result;
  }
};
}  // namespace pops::runtime::program
