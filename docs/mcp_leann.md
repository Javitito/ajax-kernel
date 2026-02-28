# MCP LEANN (Read-only)

Este servidor expone LEANN vía MCP por `stdio` usando JSON-RPC delimitado por líneas (`\n`), sin mutaciones.

## Server

Comando local:

```bash
python agency/mcp/leann_server.py
```

El server publica estas tools:

- `leann.search(query, k=5, filters={})`
- `leann.read(doc_id, span|range)`
- `leann.receipts.latest(k=20)` (opcional)

## Guardarraíles

- Read-only: no reindex ni escrituras sobre LEANN.
- Si índice/LEANN no está disponible: respuesta `ok=false`, `error=capability_missing`, `how_to_fix`.
- Sin secretos en salida (solo metadata/paths de artefactos).

## Smoke

Ejecuta:

```bash
python tools/mcp_smoke_leann.py
```

El script arranca el server, envía 1 request JSON-RPC `leann.search` y valida respuesta.

## Ejemplo request JSON-RPC

```json
{"jsonrpc":"2.0","id":"smoke-1","method":"leann.search","params":{"query":"providers audit","k":3,"filters":{}}}
```

