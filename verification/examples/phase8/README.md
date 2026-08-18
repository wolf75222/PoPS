# Phase 8 example visual artefacts

**These files are DETERMINISTIC FIXTURES. They are not live PoPS campaign results.**

They exist so the Phase 8 renderer, `visual_manifest.json` schema, and
publication/storyboard layout can be inspected without inventing a scientific
curve from a missing run. Every `status.json` sets
`data_kind = "deterministic_fixture"`. Figure captions repeat that label and
the fixture provenance SHA.

Regenerate:

```bash
python scripts/render_verification_visuals.py \
  --examples verification/examples/phase8 \
  --formats svg,png,pdf
```

Phase 8.0 acceptance pair: `TR-01` and `PO-01` (1-d and 2-d release packs).
3-d slice companions live under `TR-01/fixture-3d`. Release dashboards live
under `gallery/`. MP4/GIF masters are omitted here because a working `ffmpeg`
is required; frames and storyboards are still written when animation data
exist.

A real campaign plot must consume schema-valid `metrics.json`,
`provenance.json`, and `analysis/visual_data`. Missing source data fails
closed. Verdicts stay `pass` / `fail` / `not-supported` / `not-run`.
