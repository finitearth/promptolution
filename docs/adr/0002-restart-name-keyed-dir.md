# Restart via name-keyed output dir + override-keyed subdir

**Status:** accepted (2026-06-22)

A crashed or partial grid must resume into the **same** save folder, redoing only the missing cells. We key
the output folder on a required `experiment.name` (`hydra.sweep.dir = <root>/<name>`, **no timestamp**) and
each cell's subdirectory on its swept parameters (`${hydra.job.override_dirname}`, with sensitive/noisy keys
like `api_key` excluded and a hash fallback if the slug is long). Re-running the same experiment name skips
cells that already wrote a `.finished` marker and adds the rest in place.

## Considered options
- **Timestamped sweep dir + `job.num` subdir** (Hydra's default): rejected — every rerun lands in a new
  folder (no resume), and `job.num` shifts when the sweep is reordered or an axis value is added, so
  skip-completed would match the wrong cell.
- **User passes an explicit output dir each run**: rejected — manual and error-prone.

## Consequences
- Cell directories are self-labeling by their overrides, which also helps analysis/plotting recover which
  cell ran which config.
