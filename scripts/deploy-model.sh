#!/usr/bin/env bash
# deploy-model.sh
#
# Uploads model.onnx to Hugging Face and sets all required Vercel env vars.
#
# USAGE:
#   1. Fill in the four variables below (HF_TOKEN, VERCEL_TOKEN, VERCEL_PROJECT_ID, OPENROUTER_KEY)
#   2. Run:  bash scripts/deploy-model.sh
#
# WHERE TO GET EACH VALUE:
#   HF_TOKEN          → https://huggingface.co/settings/tokens  (create a Write token)
#   VERCEL_TOKEN      → https://vercel.com/account/tokens        (create a token)
#   VERCEL_PROJECT_ID → vercel.com → chess2pdf project → Settings → General → Project ID
#   OPENROUTER_KEY    → https://openrouter.ai/keys               (optional – leave blank to skip AI coach)

# ── FILL IN THESE FOUR VALUES ───────────────────────────────────────────────
HF_TOKEN=""
VERCEL_TOKEN=""
VERCEL_PROJECT_ID=""
OPENROUTER_KEY=""        # optional
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HF_USER="Westcoastrenmen"
HF_REPO="chess2pdf-fenify"
MODEL_FILE="$(dirname "$0")/../public/fenify/model.onnx"
MODEL_URL="https://huggingface.co/${HF_USER}/${HF_REPO}/resolve/main/model.onnx"

VERCEL_TEAM_ID=""   # leave blank if personal account; fill in if you use a Vercel team

# ── validate inputs ──────────────────────────────────────────────────────────
if [[ -z "$HF_TOKEN" || -z "$VERCEL_TOKEN" || -z "$VERCEL_PROJECT_ID" ]]; then
  echo "ERROR: Please fill in HF_TOKEN, VERCEL_TOKEN, and VERCEL_PROJECT_ID inside this script."
  exit 1
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: model.onnx not found at $MODEL_FILE"
  exit 1
fi

echo "=== Step 1: Create Hugging Face repo (if it doesn't exist) ==="
curl -sf -o /dev/null -X POST \
  "https://huggingface.co/api/repos/create" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"model\",\"name\":\"${HF_REPO}\",\"private\":false}" \
  && echo "  Repo created or already exists." \
  || echo "  (Repo may already exist — continuing)"

echo ""
echo "=== Step 2: Upload model.onnx to Hugging Face (~119 MB, please wait) ==="
UPLOAD_RESPONSE=$(curl -sf -X POST \
  "https://huggingface.co/api/models/${HF_USER}/${HF_REPO}/upload/main/model.onnx" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@${MODEL_FILE}")

echo "  Upload response: $UPLOAD_RESPONSE"
echo "  Model URL: $MODEL_URL"

echo ""
echo "=== Step 3: Verify model is publicly accessible ==="
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" --head "$MODEL_URL" || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
  echo "  ✓ Model is live at $MODEL_URL"
else
  echo "  WARNING: Got HTTP $HTTP_CODE — model may still be processing. Wait 30 seconds and retry the URL in a browser."
fi

echo ""
echo "=== Step 4: Set Vercel environment variables ==="

VERCEL_API="https://api.vercel.com"
TEAM_PARAM=""
if [[ -n "$VERCEL_TEAM_ID" ]]; then
  TEAM_PARAM="?teamId=${VERCEL_TEAM_ID}"
fi

set_vercel_env() {
  local KEY="$1"
  local VALUE="$2"
  local TARGETS='["production","preview","development"]'

  # Delete existing value first (ignore errors if not found)
  curl -sf -o /dev/null -X DELETE \
    "${VERCEL_API}/v10/projects/${VERCEL_PROJECT_ID}/env/${KEY}${TEAM_PARAM}" \
    -H "Authorization: Bearer $VERCEL_TOKEN" \
    -H "Content-Type: application/json" || true

  # Create new value
  RESULT=$(curl -sf -X POST \
    "${VERCEL_API}/v10/projects/${VERCEL_PROJECT_ID}/env${TEAM_PARAM}" \
    -H "Authorization: Bearer $VERCEL_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"key\":\"${KEY}\",\"value\":\"${VALUE}\",\"type\":\"plain\",\"target\":${TARGETS}}")

  echo "  Set $KEY → $RESULT"
}

set_vercel_env "NEXT_PUBLIC_FENIFY_MODEL_URL" "$MODEL_URL"
set_vercel_env "NEXT_PUBLIC_SITE_URL" "https://chess2pdf.vercel.app"
set_vercel_env "OPENROUTER_MODEL" "nvidia/nemotron-3-super-120b-a12b:free"

if [[ -n "$OPENROUTER_KEY" ]]; then
  # OPENROUTER_API_KEY must NOT be NEXT_PUBLIC_ — encrypted in Vercel
  RESULT=$(curl -sf -X POST \
    "${VERCEL_API}/v10/projects/${VERCEL_PROJECT_ID}/env${TEAM_PARAM}" \
    -H "Authorization: Bearer $VERCEL_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"key\":\"OPENROUTER_API_KEY\",\"value\":\"${OPENROUTER_KEY}\",\"type\":\"encrypted\",\"target\":[\"production\",\"preview\",\"development\"]}")
  echo "  Set OPENROUTER_API_KEY (encrypted) → $RESULT"
else
  echo "  Skipping OPENROUTER_API_KEY (not provided)"
fi

echo ""
echo "=== Step 5: Trigger Vercel redeploy ==="
LATEST_DEPLOY=$(curl -sf \
  "${VERCEL_API}/v6/deployments${TEAM_PARAM:+$TEAM_PARAM&}projectId=${VERCEL_PROJECT_ID}&limit=1" \
  -H "Authorization: Bearer $VERCEL_TOKEN" | python3 -c "import sys,json; d=json.load(sys.stdin)['deployments']; print(d[0]['uid']) if d else print('')" 2>/dev/null || echo "")

if [[ -n "$LATEST_DEPLOY" ]]; then
  REDEPLOY=$(curl -sf -X POST \
    "${VERCEL_API}/v13/deployments${TEAM_PARAM}" \
    -H "Authorization: Bearer $VERCEL_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"deploymentId\":\"${LATEST_DEPLOY}\",\"name\":\"chess2pdf\"}")
  echo "  Redeploy triggered: $REDEPLOY"
else
  echo "  Could not find latest deployment. Push any commit to GitHub to trigger a new deploy:"
  echo "    cd /path/to/chess2pdf"
  echo "    git commit --allow-empty -m 'chore: trigger redeploy with model env' && git push"
fi

echo ""
echo "=== DONE ==="
echo ""
echo "After ~2 minutes, open https://chess2pdf.vercel.app"
echo "Upload your Silman PDF, scan a page, and look for:"
echo "  'Position recognised by the Fenify neural network'"
echo "in the Detected boards section."
