# Pipeline runtime/memory benchmark

Host: Darwin 23.6.0, arm64 (arm)
Sentences: 304 (data/evaluation/dev.jsonl + test.jsonl), all layers requested

- p50 latency: 12.7 ms
- p95 latency: 39.2 ms
- p99 latency: 66.4 ms
- median latency: 12.6 ms
- peak traced memory during the run (tracemalloc, Python-object allocations only — does not include the Stanza model's own C/torch buffers): 1.5 MB

Caveats: single-process, single-run measurement on whatever machine this was generated on (see Host above) — not averaged across multiple runs or isolated from other system load. `tracemalloc` tracks Python heap allocations; native memory used by Stanza's underlying torch model is not captured by it, so peak memory here understates true process RSS.
