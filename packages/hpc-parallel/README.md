# hpc-parallel

Separate HPC parallelization and instrumentation package for Microsoft GraphRAG 3.3.0.

## Phase 1: Instrumentation Only

Phase 1 provides configurable execution mode definitions (`ExecutionMode`) and zero-overhead
function-level timing, concurrency, batch/request count, and error instrumentation exported
as CSV and JSON without altering default GraphRAG behavior or implementing Ray/Dask/CUDA yet.
