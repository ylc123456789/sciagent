#!/usr/bin/env bash
# One production CLI path and fixed protocol limits for every L3 command.
set -euo pipefail

usage() {
    echo 'Usage: bash e2e/l3_calibration/run.sh ABSOLUTE_ROOT run|show|answer|resume RUN_ID [ANSWER_ARGS...]' >&2
    exit 2
}

[ "$#" -ge 3 ] || usage
l3_root=$1
l3_command=$2
l3_run_id=$3
shift 3
case "$l3_root" in /*) ;; *) usage ;; esac
case "$l3_command" in run|show|answer|resume) ;; *) usage ;; esac
case "$l3_run_id" in ''|*[!a-zA-Z0-9_-]*) usage ;; esac
[ -f "$l3_root/manifest.json" ] || { echo 'Prepare a new L3 root first.' >&2; exit 2; }
[ -f "$l3_root/run-config.sh" ] || usage
[ -f "$l3_root/python-path.txt" ] || usage
IFS= read -r l3_python < "$l3_root/python-path.txt"
case "$l3_python" in /*) ;; *) usage ;; esac
[ -x "$l3_python" ] || { echo 'The prepared Python interpreter is no longer available.' >&2; exit 2; }
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
launcher=(/bin/bash --noprofile --norc "$script_root/apps/cli/run-configured.sh"
    --python "$l3_python" "$l3_root/run-config.sh")
cd -- "$l3_root"

case "$l3_command" in
    run)
        [ "$#" -eq 0 ] || usage
        "$l3_python" "$script_root/e2e/l3_calibration/prepare.py" \
            --root "$l3_root" --claim-run "$l3_run_id"
        exec "${launcher[@]}" run \
            --run-id "$l3_run_id" \
            --workspace "$l3_root/workspace" \
            --data-root "$l3_root/data" \
            --goal-file "$l3_root/goal.txt" \
            --required-artifact comparison_results \
            --constraint '只修改实验工作区，不修改框架源码或外部验收文件。' \
            --constraint '使用已有数据资源；缺少资源或需要额外权限、费用时询问用户。' \
            --execute-commands --approval-mode auto_review \
            --max-tasks 12 --max-attempts 2 --max-llm-calls 200 \
            --timeout-seconds 14400
        ;;
    show|resume)
        [ "$#" -eq 0 ] || usage
        exec "${launcher[@]}" "$l3_command" "$l3_run_id" --data-root "$l3_root/data"
        ;;
    answer)
        [ "$#" -gt 0 ] || usage
        exec "${launcher[@]}" answer "$l3_run_id" --data-root "$l3_root/data" "$@"
        ;;
esac
