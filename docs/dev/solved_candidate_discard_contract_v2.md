# SolveOutcome publication contract v2

`SolveOutcome::discard_candidate()` explicitly closes a solved staged candidate
without accepting it. `publication_contract_version` is 2. The corresponding
collective action `SolveConsumption::kDiscardCandidate` is appended at value 3;
legacy Accept/RejectAttempt/FailRun values 0/1/2 and semantics remain unchanged.
There is no class-layout, native ABI, Python IR or checkpoint-wire change.

An unsuccessful pre-Accept validation still leaves the Outcome pending and can
be retried after a reversible authority repair. An irreversibly revoked staged
lease instead needs an explicit discard: the existing solved numerical report
must not be relabelled failed merely to release its publication reservation.

Discard uses the existing owner, incarnation, external lifetime, report
disposition and collective action guards before closing. It is permitted only
for a solved report; it cannot downgrade or bypass a failed FailRun. It does not
call pre-Accept validation, Accept, reject or numerical failure hooks. It invokes
the release hook exactly once and releases the pending native attempt. A later
Accept or second discard fails. The numerical report is preserved exactly;
its solved status describes the numerical result, not candidate acceptance.
The void convenience method deliberately returns no purported accepted report.

The actual partial-refinement original-field fixture now explicitly discards
after a point mutation or lease revocation refuses Accept, then checks that all
live levels remain unchanged. Native System fixtures additionally check that
discard removes the pending-step barrier without changing state/time/step. New
serial and MPI fixture cases cover action disagreement, retryable validation,
failed-report refusal, callback separation, legacy ordinals and release once.
The included native-attempt review fragment is now explicitly registered for
serial CTest discovery, retaining all its existing cases as well as the new one.

Source-host probes compile the actual SolveOutcome and SolveReport bodies with
only scalar/communicator declarations replaced. Debug and NDEBUG runs preserve
the fail-stop destructor for an unconsumed Outcome. These host tests do not
qualify native MPI or runtime publication; native execution remains root owned.
