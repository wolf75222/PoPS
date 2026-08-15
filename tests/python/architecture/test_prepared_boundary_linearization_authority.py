"""The Uniform boundary linearization has one generated, collective authority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PROGRAM = ROOT / "src" / "runtime" / "system" / "system_program.cpp"
SYSTEM_INSTALL = ROOT / "src" / "runtime" / "system" / "system_install.cpp"
PROGRAM_CONTEXT = ROOT / "include" / "pops" / "runtime" / "program" / "program_context.hpp"
SYSTEM_BLOCK = ROOT / "include" / "pops" / "runtime" / "builders" / "compiled" / "generated_system_block.hpp"
SYSTEM_CLOSURES = ROOT / "include" / "pops" / "runtime" / "system" / "system_block_closures.hpp"
SYSTEM_STORE = ROOT / "include" / "pops" / "runtime" / "system" / "system_block_store.hpp"
SCALAR_SESSION = ROOT / "include" / "pops" / "runtime" / "program" / "prepared_scalar_boundary_session.hpp"


def _between(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_generated_uniform_block_constructs_prepared_residual_and_jvp_closures():
    source = SYSTEM_BLOCK.read_text(encoding="utf-8")
    closures = _between(
        source,
        "result.closures.rhs_core_at_point_prepared =",
        "result.closures.prepare_generated_state_at_point =",
    )

    assert "result.closures.boundary_residual_at_point_prepared" in closures
    assert "result.closures.boundary_jvp_at_point_prepared" in closures
    assert "result.closures.boundary_full_at_point_prepared" in closures
    assert "full_with_transport(boundary_state, boundary_total, &boundary, transport)" in closures
    assert "full_with_transport(core_state, core_total, nullptr, transport)" in closures
    assert "transport.fill_halo(state)" in source
    assert "HaloExchangeContext{1, 1" not in source
    assert "for (int component = 0; component < state.ncomp(); ++component)" in closures
    assert "norm_inf(state, component)" in closures
    assert "all_reduce_max(static_cast<double>(local_state_scale), lane)" in closures
    assert "norm_inf(direction, component)" in closures
    assert "all_reduce_max(static_cast<double>(local_direction_scale), lane)" in closures
    assert "/\n            direction_scale" in closures
    assert "std::sqrt(std::numeric_limits<Real>::epsilon()) * (Real(1) + state_scale)" in closures
    assert closures.count("const ExecutionLane& lane") >= 3
    assert "boundary_residual(point, state, base, boundary, lane, transport)" in closures
    assert "boundary_residual(point, perturbed, displaced, boundary, lane, transport)" in closures


def test_system_executes_only_prepared_closures_through_collective_transaction():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    residual = _between(
        source,
        "void System<Dim>::block_boundary_residual_into_at(",
        "template <int Dim>\nvoid System<Dim>::block_boundary_jvp_into_at(",
    )
    jvp = _between(
        source,
        "void System<Dim>::block_boundary_jvp_into_at(",
        "template <int Dim>\nvoid System<Dim>::block_prepare_generated_state_at(",
    )

    for body, prepared in ((residual, "boundary_residual_at_point_prepared"),
                           (jvp, "boundary_jvp_at_point_prepared")):
        assert "collective_boundary_preflight<Dim>" in body
        assert "invoke_prepared_boundary_transaction<Dim>" in body
        assert "prepared_block != block || prepared_point != point" in body
        assert "prepared_system != this" in body
        assert prepared in body
        assert "lane, transport" in body
    assert "boundary_residual_at_point(" not in residual
    assert "boundary_jvp_at_point(" not in jvp


def test_program_session_is_collectively_authenticated_and_forwards_to_system():
    source = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    session = _between(
        source,
        "std::shared_ptr<block_boundary_session_type> prepare_block_boundary_session(",
        "bool has_boundary_linearization(",
    )

    assert "all_reduce_max(local_error ? 1L : 0L, lane)" in session
    assert "all_ranks_agree_exact_ordered_byte_pairs" in session
    assert "lane.identity()" in session
    assert "lane.borrow_immutably()" in SCALAR_SESSION.read_text(encoding="utf-8")
    assert "resolve_prepared_program_block_" in source
    assert "boundary.system()" in source
    assert "boundary.transport()" in source
    assert "system_->block_boundary_residual_into_at(" in source
    assert "system_->block_boundary_jvp_into_at(" in source
    assert "boundary.require(" not in source


def test_transaction_validates_all_components_with_the_borrowed_lane_before_publication():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    transaction = _between(
        source,
        "void invoke_prepared_boundary_transaction(",
        "\n}\n\n}  // namespace",
    )

    assert "for (int component = 0; component < field.ncomp(); ++component)" in source
    assert "all_reduce_max(static_cast<double>(local_norm), lane)" in source
    validation = transaction.index("prepared_boundary_local_norm_inf(*candidate)")
    publication = transaction.index("copy_field_storage(*candidate, result)", validation)
    assert validation < publication


def test_uniform_boundary_has_no_parallel_raw_or_world_lane_runtime_engine():
    install = SYSTEM_INSTALL.read_text(encoding="utf-8")
    closures = SYSTEM_CLOSURES.read_text(encoding="utf-8")
    store = SYSTEM_STORE.read_text(encoding="utf-8")
    wrapper = _between(
        install,
        "void System<Dim>::stage_prepared_ghost_boundary_component(",
        "\ntemplate <int Dim>\nvoid System<Dim>::install_hyperbolic_boundary(",
    )

    assert "component_lane" not in wrapper
    assert "apply_ghost_region<Dim>(point, state, geometry, lane)" in wrapper
    assert "original(point, state, result, boundary, lane, transport)" in wrapper
    for source in (closures, store):
        assert "PointJvp" not in source.replace("PreparedPointJvp", "")
        assert "PointResidual boundary_residual_at_point;" not in source


def test_prepared_transport_is_collective_once_and_hot_path_reuses_halo_only():
    source = SCALAR_SESSION.read_text(encoding="utf-8")
    assert "void fill_halo(field_type& field) const" in source
    assert "fill_physical_boundary(field, *physical_)" in source
    assert "all_reduce_max(local_error ? 1L : 0L, lane.communicator())" in source
    assert "std::make_unique<HaloExchange<Dim>>(*schedule_, *lane_, context)" in source
