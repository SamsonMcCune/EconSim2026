# rust_evo_market

This project converts your Python simulation into Rust in two stages.

## Layout

- `src/lib.rs`
  - shared model code
  - discrete simulation logic
  - continuous-time world logic
  - CSV export helpers
- `src/bin/stage1_discrete.rs`
  - headless discrete-round simulation
  - writes CSV outputs to `output/discrete`
- `src/bin/stage2_nannou.rs`
  - continuous-time GUI using nannou
  - pause, reset, speed controls, CSV dump

## Commands

```bash
cargo run --release --bin stage1_discrete
cargo run --release --bin stage2_nannou
```

## What is intentionally preserved from your current Python logic

This Rust port keeps the core quirks of the original model on purpose for stage 1:

- relationship candidate search is still capped at distance `0.1`
- a pair must also satisfy `distance <= radius_a + radius_b`
- child `x` and `y` grid cells are **not** recomputed after child position is overwritten, matching the current Python behavior
- `normalize` still returns `0.5 + normalized_value`
- relationship score is still clipped into `[-1, 1]`
- marriage objects still keep their own stored `food_surplus` instead of being refreshed from the latest relationship object
- trade utility tests still compare new **expected** utility to old **true** utility

## Stage 2 design choice

True continuous time requires a modeling choice because the Python code is round-based. This version uses:

- continuous production and consumption with `dt`
- continuous survival cost integration with `dt`
- event clocks for learning, market clearing, and reproduction
- the same social and market mechanics reused as closely as possible inside those event clocks

That gives you a live continuous simulation without throwing away your original model structure.

## Output files from stage 1

- `history.csv`
- `market.csv`
- `orders.csv`
- `snapshots.csv`
- `resource_grid.csv`
- empirical supply/demand CSVs for both goods
