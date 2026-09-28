# Actual delegation

The task used callable subagents; no delegation was simulated.

| Agent | Model/profile | Work and review |
| --- | --- | --- |
| astra_semantics | gpt-6-astra/high | Program algebra, Newton captures, native intermediate guards; independent limiter/cache/path review |
| sol6_native | gpt-6-sol/high | Native limiter/WENO fixes, rollback test, scalar microbenchmark; independent prepared-resource review |
| sol6_reference | gpt-6-sol/high | Reference receipts, corpus mapping, adversarial expressions, independent three-component path and curved-path oracles |
| astra_protocols | gpt-6-astra/high | Native collective resource failure, deep freeze deletion, independent protocol evidence |
| Primary | session model | Integration, baseline, generic path lowering, installed/native qualification and final review |

Two earlier Sol 5.6 workers were interrupted after the explicit model correction.
The initial isolated reference process had already been launched; its output
was retained and checked by Sol 6. It is not attributed as code written by a
GPT-6 worker. All subsequent coding and counter-reviews above use GPT-6.

Authors did not accept their own changes alone. Sol's expression counterexample
found temporal coefficient promotion; Astra's path review found diagonal and
quadrature consistency plus missing runtime-parameter discovery; Sol found the
unrestricted integral omission. The primary corrected the path mechanism and
the independent tests were retained. Reports distinguish source-only checks
from installed/native runs.
