#!/bin/bash
# PostToolUse hook for DynaRapid vision kernel Write operations.
# Reads JSON event from stdin, extracts the file path, and runs the verifier.
#
# Claude Code passes:
#   {"tool_name": "Write", "tool_input": {"file_path": "...", "content": "..."}, ...}
#
# Exit 0 = pass / not applicable (silent)
# Exit 2 = fail (stderr shown to Claude, blocks further progress)

set -e

# Read the hook event JSON from stdin
EVENT=$(cat)

# Extract the file path (jq preferred; fall back to python)
if command -v jq >/dev/null 2>&1; then
    FILE_PATH=$(echo "$EVENT" | jq -r '.tool_input.file_path // empty')
    TOOL=$(echo "$EVENT" | jq -r '.tool_name // empty')
else
    FILE_PATH=$(echo "$EVENT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))")
    TOOL=$(echo "$EVENT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_name',''))")
fi

# Only act on Write/Edit to dataset/kernels/*_int32/generate.py
if [[ "$TOOL" != "Write" && "$TOOL" != "Edit" ]]; then
    exit 0
fi
if [[ ! "$FILE_PATH" =~ /dataset/kernels/[A-Za-z0-9_]+_int32/generate\.py$ ]]; then
    exit 0
fi

# Resolve the repo root so we can find the Python venv and skill scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> generate_dynarapid_vision_kernel/ -> skills/ -> .claude/ -> <REPO_ROOT>
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Prefer the repo venv if it exists, else system python3
if [[ -x "$REPO_ROOT/venv/bin/python3" ]]; then
    PYTHON="$REPO_ROOT/venv/bin/python3"
else
    PYTHON="python3"
fi

# Run verifier. It will exit 0 on pass, 2 on fail (failure message on stderr).
exec "$PYTHON" "$SCRIPT_DIR/verify_kernel.py" "$FILE_PATH"
