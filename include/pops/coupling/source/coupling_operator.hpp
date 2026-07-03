/// @file
/// @brief Typed COUPLING OPERATOR contract wrapping the flat coupled-source program (ADC-595).
///
/// The runtime coupling representation is the flat wire POD `CoupledSourceProgram`
/// (`include/pops/runtime/facade_options.hpp`): N inputs and N output source terms addressed by
/// (block, role) handles, a rate/param bundle, per-term postfix bytecode and an optional per-cell
/// frequency program. It is bit-stable, MPI-safe and device-clean, and `System::add_coupled_source`
/// / `AmrRuntime::add_coupled_source` already lower it into the `CoupledSourceKernel` stack machine.
///
/// What was MISSING is a TYPED contract: the program is an opaque bytecode POD, so "does this coupling
/// conserve momentum?" and "what step bound does it declare?" are not machine-visible. `CouplingOperator`
/// wraps the program with two DECLARED, inspectable contracts:
///
///  - `ConservationContract`: which roles the coupling CONSERVES (its source terms over the
///    participating blocks cancel structurally) versus which it CREATES (a legal net source, e.g.
///    ionization creating an electron/ion pair). Declared by the author, VALIDATED at registration
///    (host, fail-loud) against the actual output terms -- never inferred silently.
///  - `FrequencyBound`: the declared coupling frequency mu that yields the macro-step bound
///    dt <= cfl / mu (constant, or a per-cell program already carried by the flat POD).
///
/// The numerics are UNTOUCHED: `CouplingOperator` carries the same `CoupledSourceProgram` the flat
/// path consumes; the kernels (`CoupledSourceKernel`, `CoupledFreqKernel`) never see this header. It
/// makes a coupling INSPECTABLE as an operator (System::coupled_operators() / the AMR mirror) so a
/// Program or a runtime report can enumerate couplings as typed operators with contracts, rather than
/// reading raw bytecode.
///
/// Layer: `coupling/source/` (Stable family, next to `coupled_source_program.hpp`). It sits ABOVE the
/// `.so` ABI layer exactly like `facade_options.hpp` (it never crosses the extern "C" loader boundary),
/// and includes only the flat POD plus `<string>`/`<vector>`.

#pragma once

#include <pops/runtime/facade_options.hpp>  // CoupledSourceProgram (the wrapped wire POD)

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops {

/// DECLARED conservation contract of a coupling term-set (ADC-595). `conserved_roles` are the roles
/// whose source terms cancel over the participating blocks (sum of the per-cell contributions == 0 by
/// construction, so the total quantity is invariant); `created_roles` are the roles the coupling
/// legitimately NET-sources (ionization creating an e/i pair: density is created, not conserved). A
/// role appears in AT MOST one of the two lists. Empty both -> "unchecked" (a raw user CoupledSource
/// that declares nothing keeps its historical, unvalidated behavior).
struct ConservationContract {
  std::vector<std::string> conserved_roles;  // roles whose terms structurally cancel (net == 0)
  std::vector<std::string> created_roles;     // roles the coupling legally net-sources (net != 0 ok)

  /// True when the author declared no contract at all (both lists empty) -> validation is skipped and
  /// the coupling keeps the historical unchecked behavior.
  bool unchecked() const { return conserved_roles.empty() && created_roles.empty(); }
};

/// DECLARED frequency / stability bound of a coupling (ADC-595). The couplings apply ONCE per
/// macro-step (additive splitting), so the bound is on the macro-dt: dt <= cfl / mu. `constant_mu`
/// carries the scalar declared frequency (0 = no constant bound); `per_cell` records whether the flat
/// POD also carries a per-cell program mu(U) (freq_prog_* non-empty), reduced by MAX at each step.
struct FrequencyBound {
  double constant_mu = 0.0;  // scalar mu [1/s]; dt <= cfl / mu; 0 = no constant bound
  bool per_cell = false;      // true -> the program carries a per-cell mu(U) (freq_prog_* non-empty)
};

/// A TYPED coupling operator: the flat coupled-source program PLUS its declared, inspectable contracts
/// (ADC-595). Wraps `CoupledSourceProgram` verbatim (no ABI change); the contracts are host-only
/// metadata used at registration to validate the declaration and, afterwards, to enumerate couplings as
/// operators. `label` names the operator in the inspect view and in the step-bound diagnostics.
struct CouplingOperator {
  std::string label;
  CoupledSourceProgram program;         // inputs/outputs (block+role handles), consts, term/freq bytecode
  ConservationContract conservation;    // DECLARED; validated at registration; never inferred silently
  FrequencyBound frequency;             // DECLARED macro-step bound (constant and/or per-cell)
};

/// Read-only view of a registered coupling operator (ADC-595): its label plus its declared contracts.
/// Returned by `System::coupled_operators()` / the AMR mirror so a Program or a runtime report can
/// enumerate the couplings as typed operators without touching the raw bytecode. The program itself is
/// not exposed here (it is an internal wire POD); the inspectable surface is the label + contracts.
struct CouplingOperatorView {
  std::string label;
  ConservationContract conservation;
  FrequencyBound frequency;
};

/// Structural cancellation check of ONE role's output terms in a coupled-source program (ADC-595,
/// host-only). A role is CONSERVED when its source terms come in opposite-signed pairs: for the terms
/// targeting @p role, each term's postfix program must pair with another term's program that is exactly
/// its negation (the leg emitted by `add_pair` / the named C++ couplings). At the bytecode boundary two
/// programs are opposite iff one is the other wrapped in a single trailing `Neg` opcode (CsOp::Neg == 5)
/// -- the SAME shape `CoupledSource.add_pair` emits (loss = Neg(gain)). We only decide CONSERVATION
/// positively (a paired, balanced role); a role we cannot pair is reported as unbalanced so the declared
/// contract fails loud rather than silently passing. Symmetric to the Python `_verify_conservation`,
/// which does the richer symbolic check on the Expr tree; this is the wire-level guard.
inline bool coupling_role_terms_cancel(const CoupledSourceProgram& p, const std::string& role) {
  // Collect the per-term programs (op sequences) that target this role. prog_lens segments the
  // concatenated prog_ops in term order; a term contributes to @p role iff out_roles[t] == role.
  std::vector<std::vector<int>> term_ops;
  int off = 0;
  const int n_terms = static_cast<int>(p.out_roles.size());
  for (int t = 0; t < n_terms; ++t) {
    const int len = t < static_cast<int>(p.prog_lens.size()) ? p.prog_lens[static_cast<std::size_t>(t)]
                                                             : 0;
    if (p.out_roles[static_cast<std::size_t>(t)] == role) {
      std::vector<int> ops(p.prog_ops.begin() + off, p.prog_ops.begin() + off + len);
      term_ops.push_back(std::move(ops));
    }
    off += len;
  }
  if (term_ops.empty())
    return false;  // a conserved role must have terms; none -> the declaration is inconsistent
  if (term_ops.size() % 2 != 0)
    return false;  // an odd number of legs can never pair to a net-zero exchange
  // Greedy pairing: each program must match another that is its exact negation (a trailing Neg opcode,
  // CsOp::Neg == 5). We do not depend on prog_args here: the terms of a conservative pair share the SAME
  // register program, one negated, so the op sequence alone decides (CSE keeps identical subtrees).
  constexpr int kNegOp = 5;  // CsOp::Neg (coupled_source_program.hpp, frozen ABI value)
  std::vector<bool> used(term_ops.size(), false);
  for (std::size_t i = 0; i < term_ops.size(); ++i) {
    if (used[i])
      continue;
    bool paired = false;
    for (std::size_t j = i + 1; j < term_ops.size(); ++j) {
      if (used[j])
        continue;
      const std::vector<int>& a = term_ops[i];
      const std::vector<int>& b = term_ops[j];
      // b is +Neg of a: b == a followed by a single Neg opcode.
      const bool b_is_neg_a =
          b.size() == a.size() + 1 && std::equal(a.begin(), a.end(), b.begin()) && b.back() == kNegOp;
      // a is +Neg of b: a == b followed by a single Neg opcode.
      const bool a_is_neg_b =
          a.size() == b.size() + 1 && std::equal(b.begin(), b.end(), a.begin()) && a.back() == kNegOp;
      if (b_is_neg_a || a_is_neg_b) {
        used[i] = used[j] = true;
        paired = true;
        break;
      }
    }
    if (!paired)
      return false;
  }
  return true;
}

/// Validate a coupling operator's DECLARED conservation contract against its actual terms (ADC-595,
/// host, fail-loud). Called at registration BEFORE the coupling is stored, so a coupling whose
/// declaration is inconsistent with its bytecode raises and leaves NO partial state. Checks:
///  - a role is not declared BOTH conserved and created;
///  - every declared conserved / created role actually appears in the program's out_roles;
///  - each conserved role's terms structurally cancel (opposite-signed leg pairs).
/// An empty contract (`unchecked()`) is a no-op -- a raw user CoupledSource that declares nothing keeps
/// its historical behavior. @p where names the caller in the error (e.g. "System::add_coupled_source").
inline void validate_coupling_contract(const CouplingOperator& op, const std::string& where) {
  const ConservationContract& c = op.conservation;
  if (c.unchecked())
    return;
  const std::vector<std::string>& out_roles = op.program.out_roles;
  auto role_present = [&](const std::string& r) {
    return std::find(out_roles.begin(), out_roles.end(), r) != out_roles.end();
  };
  // A role cannot be both conserved and created.
  for (const std::string& r : c.conserved_roles) {
    if (std::find(c.created_roles.begin(), c.created_roles.end(), r) != c.created_roles.end())
      throw std::runtime_error(where + " : coupling '" + op.label + "' declares role '" + r +
                               "' as BOTH conserved and created (a role is one or the other)");
    if (!role_present(r))
      throw std::runtime_error(where + " : coupling '" + op.label + "' declares conserved role '" + r +
                               "' but no source term targets it");
    if (!coupling_role_terms_cancel(op.program, r))
      throw std::runtime_error(
          where + " : coupling '" + op.label + "' declares role '" + r +
          "' conserved but its source terms do not cancel (each +expr on one block must be balanced by "
          "-expr on another; use add_pair or declare the role created)");
  }
  for (const std::string& r : c.created_roles) {
    if (!role_present(r))
      throw std::runtime_error(where + " : coupling '" + op.label + "' declares created role '" + r +
                               "' but no source term targets it");
  }
}

}  // namespace pops
