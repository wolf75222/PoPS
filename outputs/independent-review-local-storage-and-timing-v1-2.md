# Local storage follow-up and timing v1.2: independent receipt check

This review made no native rebuild or new performance measurement. It checks source tests at the frozen `PoPS-principal-group` HEAD `934206a` and reads the existing workspace receipt `outputs/performance-t2-v1-2-run/result.json`.

## Local storage

The three affected source suites (`test_local_state_storage_loader.py`, `test_source_state_storage.py`, `test_user_joint_reconstruction_storage.py`) passed independently: **20/20 in 10.82 s** using `pops-api040-c11/bin/python`, `env -u PYTHONPATH`, and `-o pythonpath=python`. This directly rechecks the multi-StateSpace A→B→A selection repair and complete System/AMR loader source generation. The checkout also has unrelated T3 working-tree edits; no test exercised a rebuilt installed native package. Installed T3/H05 acceptance is still the central campaign's task.

## Exact scope of the timing receipt

The v1.2 receipt says `status=measured` for the baseline **3d06cabee9db4a31c4d07fa4e00dc5a40d164155** (native SHA-256 `4574ed6096650aa708ad040fd6ee056414a3cfa1735a0440bd74220170265beb`) against candidate **cf6daceaa028df1343a303751ec9ae8f9f5ce6b3** (native SHA-256 `4b458dabd54357b8aa3a4659cee8363a9a4510e55aaa81706bfe03a445d27037`). It does not compare the current MAIN or `934206a` source.

The one measured scenario is a Dim2, N=48, width-5 scalar advection calculation with x=.7, y=-.4, User slope=.2, Rusanov, dt=1e-4 and 12 steps. The worker order is baseline–candidate–candidate–baseline, with two warmups and five timed samples per worker. Despite the receipt key `step_median_s`, the code times one entire `pops.run(..., max_steps=12)` call per sample and does **not** divide by 12. Median **12-step run** duration is 0.0531579375 s versus 0.0519979795 s; candidate/baseline = **0.9781790255** (about 2.18% lower for this sample). The MADs are 0.000363729 and 0.000533188 s for those whole-run durations. One baseline worker has a 0.080843 s high sample; reporting the median rather than a mean is appropriate, but the four-worker campaign cannot establish a general performance distribution.

The receipt marks numerical equivalence passed at atol=rtol=1e-12; all four worker final-state hashes are `8cf7f5e984c443de842ad855afc46a8b61f8cf75af35b9818f279a11d3a31b2f`. Cold compile entries carry separate native/headers/doctor/DSO provenance; compile times are 15.300854958 s and 15.60551175 s. Those cold compile values are not the timed run ratio. The receipt does not provide a symmetric native allocation/kernel/communication profiler, so the timing cannot establish a resource mechanism or attribute the difference uniquely to one code change. The separate v1.2 resource companion failed before profiling and has no resource result.
