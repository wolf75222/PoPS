# Independent relative quantity receipt reception

Production reviewed: `99d651c182d24aba5aaaf431c5e667f810ad5bb5`.
Private checkout: `work/PoPS-sol61-ale-quantity-codec-review`.
No production file, installed package, SDK or environment is changed by this review.

The native diagnostic recorded two one-bit disagreements in the accepted
exchange quantity. The arithmetic proof and authenticated log pins are preserved
in `28dfbb28b927a2277fefd3d7354cad73d465766f`. This review uses both operand
triples and powers-of-two component scalings; exact rational arithmetic proves
all six literal fused amounts independently of the production helper.

The new native `.inc` fixture constructs actual three-component MultiFab and
FaceField buffers, moving endpoints, exact endpoint measures, sweeps, receipts
and AcceptedExchangeLedger occurrences. Its accepted state is obtained from
Reynolds balance using literal expected face quantities. It does not call the
production relative-amount helper to construct its ledger or accepted state.
Every reader corruption test first requires an authentic positive roundtrip.

The four C++ cases check:

- Fused amounts match literal binary64 bits for both component orderings; an
  ordinary amount differing by one bit at one occurrence is refused by the exact
  receipt/ledger check, with its specific diagnostic.
- Unknown arithmetic tags 2 and UINT64_MAX are refused; failed read leaves the
  authentic image unchanged. Capture refuses wire version 3 with fused math and
  an unknown geometry wire version.
- Noninteger states and face densities with zero sweep roundtrip through
  POPSEX03 byte-exactly. The independent wrapper writes the historical call
  order without an arithmetic tag, using the existing primitive field/ledger
  encoders. Decoded legacy version/convention remain 3/0 on reexport. Initial
  historical geometry without a receipt also exports 03.
- Two retained geometry histories, one legacy and one fused, export 04 with
  different per-receipt conventions and reexport byte-exactly after decode.

The zero-sweep legacy positive deliberately avoids compiler contraction
ambiguity. It proves preservation of that historical encoding; it does not
prove that every historical POPSEX03 quantity produced by every compiler has
one portable arithmetic interpretation. Convention 0 retains the old C++
expression. Convention 1 specifies std::fma. No reader accepts an OR of the two
values or an ULP tolerance.

Source review confirmed the 03/04 reader dispatch, tag bounds, new producer's
forced version 4, and exact collective metadata for both geometry wire version
and receipt convention. Python checkpoint preflight recognizes both 03 and 04.
No new blocker was demonstrated in these reviewed seams.

Author-side checks: 8 Python source/math checks PASS (no PoPS import); Ruff PASS;
diff check PASS. A small Dim1 `-fsyntax-only` TU compiled the actual fixture
struct and all four new C++ bodies against the real private headers and Kokkos
include prefix. Assertion macros in that syntax-only TU were type-check stubs:
this was neither GTest execution nor a native runtime reception. Root must
compile and execute the real `test_program_runtime` target, including the `.inc`,
before claiming native codec reception. No executable or library was built.
