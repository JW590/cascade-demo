# CASCADE simulator explorer

An English static demo of GPT-OSS-120B decode on the CASCADE chiplet architecture.
GitHub Pages address: https://jw590.github.io/cascade-demo/

## Explore the demo

1. **Architecture:** inspect the physical 4×4 network, TP4 / PP4 groups and chiplet resources.
2. **Operators:** choose a PP stage and layer, then follow inputs, operation and outputs with tensor shapes. Communication badges identify saved D2D/Host messages. Host/PP input and TP input broadcast are separate clickable steps inside Stage input.
3. **Simulation:** use the same operator graph to jump to an operator's resource timeline. View selects Local / Communication, Operator selects a concrete operation, and Part selects all activity / communication only. Full communication views retain all saved local PE activity in the window. Inspect a service for exact operands, shapes, dependencies and resource waits; Operator graph ↑ returns to the graph.

All controls are manual. The page reads a saved run; it does not rerun the simulator.

## Open locally

Open [index.html](index.html) directly, or run:

```bash
python3 -m http.server 8861 --bind 127.0.0.1
```

Then open http://127.0.0.1:8861. Keep all seven runtime files together, including `data.js` and `run.json.gz`.

## GitHub Pages

Pages publishes the root of `main`. The empty `.nojekyll` file disables Jekyll processing. Future commits pushed to `main` update the demonstration.

## Saved simulation

The exported snapshot includes 36 layers, 107,336 commands and 108,527 services.
The estimated decode latency is 15.560307 ms under uncalibrated coarse descriptor parameters. The run uses sink-aware FlashAttention, synthetic expert assignments, warm resident weights/KV and FP16 storage.

This system model does not include host embeddings/sampling, energy, native chiplet execution, expert parallelism or multiple devices. Tensor shapes are logical operands; physical scratch tiles are not expanded.

Full-trace SHA-256: `d954002878d01af43fdd91a00a0ab625f56bbc0ae588f4ea987e9fa0d08c3fd8`.

To refresh this site, generate an updated export in the simulator project and copy all seven files from its `demo/dist/` directory into this repository. Keep generated data and the trace tracked in this presentation repository.
