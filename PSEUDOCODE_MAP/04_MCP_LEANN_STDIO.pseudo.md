# 04 - LEANN MCP Server (stdio, read-only)

## Scope
Servidor MCP local por stdio (JSON-RPC 2.0, newline delimited, lineas compactas).

## Startup

```text
main():
  backend = LeannBackend.from_env()
  print {"jsonrpc":"2.0","method":"server.ready","params":{name,version,ts_utc}}
  run_forever()
```

## Request loop

```text
run_forever():
  for line in stdin:
    if blank: continue
    parse json
    if parse fails -> error -32700
    if jsonrpc != "2.0" or method missing -> error -32600
    dispatch method
    print one-line json response + "\n"
```

## Tools

```text
leann.search(query, k=5, filters={}):
  require query non-empty
  require index meta + passages exists
  use query_leann(index_base, query, top_k=k, fallback_grep=true)
  return {ok, query, k, filters, results[]}

leann.read(doc_id, span|range):
  require doc_id
  require index exists
  find document in passages
  apply span/range slicing
  return {ok, doc_id, content, span, metadata}

leann.receipts.latest(k=20):
  list latest artifacts/receipts/*.json by mtime
  return metadata-only rows
```

## Failure contract

```text
if LEANN/index missing:
  return {
    ok: false,
    error: "capability_missing",
    operation: "<method>",
    how_to_fix: [...]
  }
```

## Invariants
- Read-only real: no auto-indexing in request path.
- No crash on missing backend; respond with stable schema.
- No embedded newlines in JSON responses.
