#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$(basename "${ROOT_DIR}")" != "ajax-kernel" ]]; then
  echo "BLOCKED: demo gate must run from ajax-kernel root. detected=${ROOT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${ROOT_DIR}/AGENTS.md" || ! -f "${ROOT_DIR}/bin/ajaxctl" || ! -d "${ROOT_DIR}/agency" ]]; then
  echo "BLOCKED: invalid runtime root markers under ${ROOT_DIR}" >&2
  exit 2
fi

cd "${ROOT_DIR}"

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TS_LABEL="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ROOT_DIR}/artifacts/reports/demo_gate/${TS_LABEL}"
LATEST_MD="${ROOT_DIR}/artifacts/reports/latest_demo.md"
RUN_MD="${RUN_DIR}/demo_report.md"
STATUS_TSV="${RUN_DIR}/status.tsv"

mkdir -p "${RUN_DIR}"

STEPS=(
  "doctor_drivers|python bin/ajaxctl doctor drivers"
  "doctor_anchor|python bin/ajaxctl doctor anchor --rail lab"
  "doctor_bridge|python bin/ajaxctl doctor bridge"
  "doctor_providers|python bin/ajaxctl doctor providers --roles brain council vision --explain"
  "health|python bin/ajaxctl health --json"
  "compileall|python -m compileall -q ."
  "tests|pytest -q tests/test_kernel_smoke.py tests/test_kernel_policy_contract.py tests/test_kernel_anchor_guard.py tests/test_kernel_preflight_fail_closed.py tests/test_kernel_health_ttl.py tests/test_kernel_ask_user_path.py"
)

: > "${STATUS_TSV}"
FAIL_STEPS=()

run_step() {
  local step_name="$1"
  local step_cmd="$2"
  local log_path="${RUN_DIR}/${step_name}.log"
  local start_ts end_ts duration rc status
  local timeout_sec="${DEMO_GATE_TIMEOUT_SEC:-180}"

  start_ts="$(date +%s)"
  if timeout --foreground "${timeout_sec}" bash -lc "${step_cmd}" >"${log_path}" 2>&1; then
    rc=0
    status="PASS"
  else
    rc=$?
    if [[ "${rc}" -eq 124 ]]; then
      status="TIMEOUT"
    else
      status="FAIL"
    fi
    FAIL_STEPS+=("${step_name}")
  fi
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))
  printf "%s\t%s\t%s\t%s\t%s\n" "${step_name}" "${rc}" "${status}" "${duration}" "${log_path}" >>"${STATUS_TSV}"
}

for item in "${STEPS[@]}"; do
  step_name="${item%%|*}"
  step_cmd="${item#*|}"
  run_step "${step_name}" "${step_cmd}"
done

remediation_for() {
  local step_name="$1"
  case "${step_name}" in
    doctor_drivers)
      echo "- Revisa puertos/sesiones con: python bin/ajaxctl doctor drivers"
      echo "- Si hay mismatch de owner/port, corrige tasks/driver y reintenta."
      ;;
    doctor_anchor)
      echo "- Ejecuta: python bin/ajaxctl doctor anchor --rail lab"
      echo "- Si sale BLOCKED, corrige session/port/display y reintenta la misma demo."
      ;;
    doctor_bridge)
      echo "- Revisa artifact de bridge doctor en artifacts/doctor."
      echo "- Verifica provider bridge y credenciales del provider CLI."
      ;;
    doctor_providers)
      echo "- Ejecuta: python bin/ajaxctl doctor providers --roles brain council vision --explain"
      echo "- Abre artifacts/health/providers/doctor_*.json y corrige auth/quota."
      ;;
    health)
      echo "- Health stale/no-green: refresca providers_status (doctor providers) y reintenta health."
      ;;
    compileall)
      echo "- Corrige errores de sintaxis/import y valida con python -m compileall -q ."
      ;;
    tests)
      echo "- Ejecuta pytest del step tests y corrige los casos fallidos."
      ;;
    *)
      echo "- Revisar log del step y aplicar fix puntual."
      ;;
  esac
}

{
  echo "# AJAX Demo Gate"
  echo ""
  echo "- ts_utc: ${TS_UTC}"
  echo "- runtime_root: ${ROOT_DIR}"
  echo "- run_dir: ${RUN_DIR}"
  echo ""
  echo "## Resumen"
  echo "| step | rc | status | duration_s | log |"
  echo "|---|---:|---|---:|---|"
  while IFS=$'\t' read -r step_name rc status duration log_path; do
    [[ -z "${step_name}" ]] && continue
    echo "| ${step_name} | ${rc} | ${status} | ${duration} | ${log_path} |"
  done <"${STATUS_TSV}"

  if [[ ${#FAIL_STEPS[@]} -eq 0 ]]; then
    echo ""
    echo "## Resultado"
    echo "Demo gate PASS."
  else
    echo ""
    echo "## Que fallo"
    for step_name in "${FAIL_STEPS[@]}"; do
      echo ""
      echo "### ${step_name}"
      echo '```text'
      tail -n 40 "${RUN_DIR}/${step_name}.log" || true
      echo '```'
    done
    echo ""
    echo "## Que hacer"
    for step_name in "${FAIL_STEPS[@]}"; do
      echo ""
      echo "### ${step_name}"
      remediation_for "${step_name}"
    done
  fi
} >"${RUN_MD}"

cp "${RUN_MD}" "${LATEST_MD}"

echo "demo_report=${RUN_MD}"
echo "latest_report=${LATEST_MD}"

if [[ ${#FAIL_STEPS[@]} -eq 0 ]]; then
  exit 0
fi
exit 2
