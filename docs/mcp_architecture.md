# MCP Architecture

The MCP surface lives inside the same installable package as the CLI.

## Ownership

- `src/contentflow/primitives/` holds the reusable schemas and deterministic media primitives.
- `src/contentflow/server.py` exposes those primitives as MCP tools.
- `src/contentflow/flows/long_to_shorts/` is a consumer of those primitives, not a separate package.

## Mental model

`contentflow` has one package and two layers:

1. `contentflow.primitives`
   Deterministic building blocks, shared schemas, and the MCP server surface.
2. `contentflow.flows.*`
   User-facing workflows that orchestrate those primitives into an end-to-end job.

Today the only shipped workflow is `contentflow.flows.long_to_shorts`.

## Entry points

- CLI: `contentflow`
- MCP server: `contentflow-mcp`

See also [`README.md`](../README.md) and [`PIPELINE.md`](PIPELINE.md).
