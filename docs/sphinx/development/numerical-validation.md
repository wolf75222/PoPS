# Numerical validation

A change to a solver, flux, Poisson solver, AMR path, backend or the DSL must show that the numbers
are still right. A unit test that only checks that the code runs is not enough.

## What to report

For such a change, give the figures so a reviewer can tell a normal difference from a silent model
change:

- the reference case;
- the observed quantity (for example a growth rate or a conserved mass);
- the expected value;
- the tolerance and why it is set there;
- the measured difference.

## Kinds of validation

- Parity: two paths that must agree exactly are compared bit for bit (native bricks against the DSL,
  one backend against another, single rank against MPI).
- Conservation: a conserved quantity (mass, momentum) is tracked over a run.
- Benchmark: a result is compared to a published or analytic target. The diocotron reproduction of
  the Hoffart case is one such benchmark; the case lives in
  [adc_cases](https://github.com/wolf75222/adc_cases).

A numerical pull request that touches a tolerance or a benchmark sources its numbers; see the
[documentation style guide](documentation.md) for the rule.
