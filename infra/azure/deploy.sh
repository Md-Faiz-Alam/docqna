#!/usr/bin/env bash
# Blue/green deploy of one backend image to Azure Container Apps.
#
#   infra/azure/deploy.sh ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
#
# 1. Switch the app to multiple-revision mode (once).
# 2. Pin 100% of traffic to the revision serving now, so the new one starts at 0%.
# 3. Create a revision with the new image.
# 4. Smoke-test it at its own address (scripts/smoke_test.py) while users
#    still get the old revision.
# 5. Pass: move 100% of traffic to it. The previous revision stays active at
#    0% for an instant rollback, and older ones are deactivated.
#    Fail: deactivate the new revision. Users never saw it.
#
# Run by .github/workflows/publish-image.yml after CI passes, or by hand in
# Azure Cloud Shell from a clone of the repository. Needs `az` signed in.
# Rollback: infra/azure/README.md.

set -euo pipefail

IMAGE="${1:?usage: deploy.sh <image>}"
APP="${APP:-docqna-api}"
RESOURCE_GROUP="${RESOURCE_GROUP:-docqna-rg}"
WEBSITE_ORIGIN="${WEBSITE_ORIGIN:-https://docqna-lake.vercel.app}"
# Lower-case letters, digits and '-', starting with a letter.
REVISION_SUFFIX="${REVISION_SUFFIX:-v$(date -u +%Y%m%d%H%M%S)}"
SMOKE_TEST="$(cd "$(dirname "$0")/../.." && pwd)/scripts/smoke_test.py"

app_args=(--name "$APP" --resource-group "$RESOURCE_GROUP")
new="$APP--$REVISION_SUFFIX"

step() { echo; echo "==> $*"; }

step "Image: $IMAGE"

mode=$(az containerapp show "${app_args[@]}" --query properties.configuration.activeRevisionsMode -o tsv)
if [ "$mode" != "Multiple" ]; then
  step "Switching $APP from $mode to multiple-revision mode"
  az containerapp revision set-mode "${app_args[@]}" --mode multiple -o none
fi

current=$(az containerapp revision list "${app_args[@]}" \
  --query "[?properties.trafficWeight > \`0\`] | [0].name" -o tsv)
if [ -z "$current" ]; then
  echo "No revision is serving traffic; refusing to guess which one is live." >&2
  exit 1
fi
step "Live revision: $current. Pinning 100% of traffic to it"
az containerapp ingress traffic set "${app_args[@]}" --revision-weight "$current=100" -o none

step "Creating revision $new at 0% traffic"
az containerapp update "${app_args[@]}" --image "$IMAGE" --revision-suffix "$REVISION_SUFFIX" -o none

state=""
for _ in $(seq 1 60); do
  state=$(az containerapp revision show "${app_args[@]}" --revision "$new" \
    --query properties.provisioningState -o tsv)
  case "$state" in
    Provisioned) break ;;
    Failed) break ;;
  esac
  sleep 10
done
if [ "$state" != "Provisioned" ]; then
  echo "Revision $new did not provision (state: ${state:-unknown})." >&2
  az containerapp revision deactivate "${app_args[@]}" --revision "$new" -o none || true
  exit 1
fi

revision_fqdn=$(az containerapp revision show "${app_args[@]}" --revision "$new" --query properties.fqdn -o tsv)
step "Smoke-testing https://$revision_fqdn (users are still on $current)"
if ! python3 "$SMOKE_TEST" "https://$revision_fqdn" --origin "$WEBSITE_ORIGIN"; then
  step "Smoke test failed: deactivating $new. Traffic stays on $current"
  az containerapp revision deactivate "${app_args[@]}" --revision "$new" -o none
  exit 1
fi

step "Moving 100% of traffic to $new ($current stays active at 0% for rollback)"
az containerapp ingress traffic set "${app_args[@]}" --revision-weight "$new=100" "$current=0" -o none

for old in $(az containerapp revision list "${app_args[@]}" \
  --query "[?properties.active && name != '$new' && name != '$current'].name" -o tsv); do
  step "Deactivating old revision $old"
  az containerapp revision deactivate "${app_args[@]}" --revision "$old" -o none
done

app_fqdn=$(az containerapp show "${app_args[@]}" --query properties.configuration.ingress.fqdn -o tsv)
step "Checking the public address https://$app_fqdn"
python3 "$SMOKE_TEST" "https://$app_fqdn" --quick

echo
echo "Deployed $IMAGE as $new."
echo "Roll back with:"
echo "  az containerapp ingress traffic set --name $APP --resource-group $RESOURCE_GROUP --revision-weight $current=100 $new=0"
