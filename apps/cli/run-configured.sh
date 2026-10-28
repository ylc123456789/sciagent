#!/usr/bin/env bash
# Use the same deployment environment for run, show, answer and resume.
set -eu

usage() {
    echo "Usage: bash run-configured.sh [--python /absolute/python] CONFIG COMMAND [ARG...]" >&2
    exit 2
}

python_executable=
if [ "${1-}" = "--python" ]; then
    [ "$#" -ge 4 ] || usage
    python_executable=$2
    shift 2
    case "$python_executable" in
        /*) ;;
        *) echo "--python requires an absolute interpreter path" >&2; exit 2 ;;
    esac
fi
[ "$#" -ge 2 ] || usage

config_file=$1
shift
case "$config_file" in
    /*) ;;
    *) config_file="$PWD/$config_file" ;;
esac

set -a
source "$config_file"
set +a
if [ -n "$python_executable" ]; then
    exec "$python_executable" -m sciagent_cli "$@"
fi
exec sciagent "$@"
