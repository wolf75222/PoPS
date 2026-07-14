#pragma once

#include <string>
#include <vector>

/// @file
/// @brief OPTIONS PODs for the public facades (System / AmrSystem), grouping the long families of
///        HOMOGENEOUS parameters that posed an ordering footgun (C++ Core Guidelines I.23).
///
/// Layer: `include/pops/runtime`.
/// Role: carry the bytecode description of an inter-species coupled source. This family was previously
///   a flat list of parameters of the SAME type (several parallel
///   `std::vector<int>`) -- silently swappable at the call site. Grouping them into a named POD makes
///   the call self-documenting (designated initializers) and removes the swap risk.
/// Contract: flat POD crossing the bindings without friction. The in-class DEFAULTS reproduce EXACTLY
///   the old defaults of the flat parameters -> no behavior change.
///
/// Invariants:
/// - these PODs live ABOVE the authenticated component ABI (`native_loader.hpp`): they never
///   cross the extern "C" boundary of a component loader. The SEMANTIC extern "C" ABI (residual / advance,
///   structs crossing the loader) therefore stays UNCHANGED. On the other hand the abi_key() LITERAL
///   CHANGES: it embeds the token headers=POPS_HEADER_SIG (conservative sha256 of the path and content
///   of EVERY header under include/, cf. abi_key.hpp and python/CMakeLists.txt); merely ADDING this
///   header and EDITING system.hpp / amr_system.hpp shifts POPS_HEADER_SIG. This is EXPECTED and
///   harmless: no semantic ABI changes, but add_native_block will reject the native component generated before
///   this change (divergent signature) -> a one-time regeneration of the stale .so.

namespace pops {

/// @brief BYTECODE description of a generic inter-species COUPLED SOURCE (cf.
///        System::add_coupled_source / AmrSystem::add_coupled_source). Groups the FLAT arrays
///        of the bytecode ABI -- six `std::vector` (four of block/role descriptors, two+ of stack
///        machine program) swappable at the call site -- into a single named aggregate.
///
/// Usage: built by the facade (or by the bindings, from the flat Python kwargs) then passed to
///   add_coupled_source with the frequency and the label kept flat (a double and a string, distinct
///   types, outside the homogeneous footgun). A malformed shape raises an EXPLICIT error on add.
/// Contract: FLAT ABI -- no C++ object crosses the boundary; this POD is only a facade-side carrier of
///   arrays. The DEFAULTS reproduce the old defaults of the flat parameters (the EMPTY per-cell
///   frequency programs = constant frequency alone, bit-identical).
///  - in_blocks / in_roles: blocks read as input and their roles (one per input register).
///  - consts: constants (.param()), loaded after the inputs.
///  - out_blocks / out_roles: target block and target role of each source term.
///  - prog_ops / prog_args: concatenated opcodes of ALL terms (stack machine) and their parallel
///    arguments (register index for PushReg).
///  - prog_lens: program length of each term (segments prog_ops / prog_args in order).
///  - freq_prog_ops / freq_prog_args: OPTIONAL program of a PER-CELL frequency mu(U) (same stack
///    machine, SAME register table). EMPTY (default) = constant frequency alone.
struct CoupledSourceProgram {
  std::vector<std::string> in_blocks;
  std::vector<std::string> in_roles;
  std::vector<double> consts;
  std::vector<std::string> out_blocks;
  std::vector<std::string> out_roles;
  std::vector<int> prog_ops;
  std::vector<int> prog_args;
  std::vector<int> prog_lens;
  std::vector<int> freq_prog_ops;
  std::vector<int> freq_prog_args;
};

}  // namespace pops
