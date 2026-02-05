#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def list_runs(n: int = 10) -> List[Path]:
    runs = sorted([p for p in (ROOT / 'runs').glob('*') if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[:n]


def main() -> int:
    print('# Ajax Dashboard\n')
    # Últimos runs (heurístico)
    print('## Últimos runs')
    for p in list_runs(8):
        status = '✅'
        if (p / 'result.json').exists():
            try:
                res = load_json(p / 'result.json')
                status = '✅' if res.get('ok') else '❌'
            except Exception:
                status = '⚠️'
        print(f'- {status} {p.name}')

    # Cuotas y latencias
    quota = load_json(ROOT / 'artifacts' / 'providers_quota.json')
    print('\n## Proveedores (quota)')
    prov = quota.get('providers') or {}
    for name, info in prov.items():
        print(f"- {name}: ok_to_use={info.get('ok_to_use')} margin={info.get('margin')}")

    # Próximo nightly (informativo)
    print('\n## Próximo nightly')
    print('Job: jobs/nightly_refresh.json (catálogo → bench → routing → KPIs)')
    print('- Self-audit semanal (cron sugerido): 0 5 * * 0 python3 scripts/self_audit_weekly.py')
    print('- Último self-audit: artifacts/ajax_self_audit.md')

    # Acciones rápidas (comandos)
    print('\n## Acciones rápidas')
    print('- Ver PoC LangFlow: python3 agency/broker.py jobs/poc_langflow.json --print-result')
    print('- Relanzar MetaGPT: python3 agency/broker.py jobs/poc_metagpt.json --print-result')
    print('- Abrir comparativo (cuando esté): artifacts/poc_compare.md')
    # PoC semáforos
    print('\n## PoC · Semáforos')
    def _poc_line(name: str, label: str, file: str, lic_file: str, is_rag: bool = False) -> None:
        m = load_json(ROOT / 'artifacts' / file)
        lic = (load_json(ROOT / 'artifacts' / lic_file).get('risk') or 'unknown')
        p95 = int(m.get('latency_p95_ms', 0))
        okr = float(m.get('pass_ratio', 0))
        cost = float(m.get('cost_est_eur', 0))
        wh = float(m.get('energy_wh', 0))
        limit = 8000 if is_rag else 5000
        go = (okr >= 0.8 and p95 <= limit and cost <= 0.5 and wh <= 1.5 and lic not in {'high','copyleft'})
        badge = '🟢 GO' if go else '🟠 PARK'
        print(f"- {label}: {badge} · p95={p95} ok={okr:.2f} €={cost:.4f} Wh={wh:.4f} lic={lic}")
    _poc_line('langflow','LangFlow','poc_langflow_metrics.json','poc_langflow_license.json', False)
    _poc_line('metagpt','MetaGPT','poc_metagpt_metrics.json','poc_metagpt_license.json', True)
    _poc_line('openbb','OpenBB','poc_openbb_metrics.json','poc_openbb_license.json', False)
    print('\nEnlace plan 2 semanas: artifacts/poc_compare.md')
    print('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
