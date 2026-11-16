#!/usr/bin/env bash
# Deploy one backend image: staging first, then production.
#
#   infra/azure/deploy.sh ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
#
# 1. Update the staging app (docqna-api-staging) to the new image.
# 2. Full smoke test against staging (scripts/smoke_test.py), once /info
#    reports the new commit. Fail: stop here. Production is never touched.
# 3. Update production to the same image. Azure moves traffic to the new
#    revision only once it has started; until then the old one serves.
# 4. Quick smoke test against production. Fail: redeploy the previous
#    production image (automatic rollback) and fail the run.
#
# The apps run in an Azure Container Apps Express environment, which allows
# one active revision per app, so a separate staging app takes the place of
# testing a new revision at 0% traffic.
#
# Run by .github/workflows/publish-image.yml after CI passes, or by hand in
# Azure Cloud Shell from a clone of the repository. Needs `az` signed in.

set -euo pipefail

IMAGE="${1:?usage: deploy.sh <image>}"
BUILD="${BUILD_SHA:-${IMAGE##*:}}"
APP="${APP:-docqna-api}"
STAGING_APP="${STAGING_APP:-docqna-api-staging}"
RESOURCE_GROUP="${RESOURCE_GROUP:-docqna-rg}"
WEBSITE_ORIGIN="${WEBSITE_ORIGIN:-https://docqna-lake.vercel.app}"
# Lower-case letters, digits and '-', starting with a letter.
REVISION_SUFFIX="${REVISION_SUFFIX:-v$(date -u +%Y%m%d%H%M%S)}"
SMOKE_TEST="$(cd "$(dirname "$0")/../.." && pwd)/scripts/smoke_test.py"

step() { echo; echo "==> $*"; }

fqdn() {
  az containerapp show --name "$1" --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv
}

image_of() {
  az containerapp show --name "$1" --resource-group "$RESOURCE_GROUP" \
    --query "properties.template.containers[0].image" -o tsv
}

deploy_to() {  # app image revision-suffix
  az containerapp update --name "$1" --resource-group "$RESOURCE_GROUP" \
    --image "$2" --revision-suffix "$3" -o none
}

step "Image: $IMAGE (build $BUILD)"

if ! az containerapp show --name "$STAGING_APP" --resource-group "$RESOURCE_GROUP" -o none 2>/dev/null; then
  echo "The staging app $STAGING_APP does not exist. Create it once: infra/azure/README.md" >&2
  exit 1
fi

# --- Staging -----------------------------------------------------------------

staging_url="https://$(fqdn "$STAGING_APP")"
step "Deploying to staging ($STAGING_APP)"
deploy_to "$STAGING_APP" "$IMAGE" "$REVISION_SUFFIX"

step "Full smoke test on staging: $staging_url"
if ! python3 "$SMOKE_TEST" "$staging_url" --origin "$WEBSITE_ORIGIN" --expect-build "$BUILD" --wait 600; then
  step "Staging failed. Production was not touched."
  exit 1
fi

# --- Production --------------------------------------------------------------

previous=$(image_of "$APP")
production_url="https://$(fqdn "$APP")"
step "Staging passed. Deploying to production ($APP); previous image: $previous"
deploy_to "$APP" "$IMAGE" "$REVISION_SUFFIX"

step "Quick smoke test on production: $production_url"
if python3 "$SMOKE_TEST" "$production_url" --quick --origin "$WEBSITE_ORIGIN" --expect-build "$BUILD" --wait 600; then
  echo
  echo "Deployed $IMAGE to staging and production."
  echo "Roll back with:"
  echo "  az containerapp update --name $APP --resource-group $RESOURCE_GROUP --image $previous"
  exit 0
fi

# Only an image pinned to a commit is a safe rollback target: a moving tag
# such as :latest may already point at the image that just failed.
if [[ "$previous" == *:latest || "$previous" != *:* ]]; then
  step "Production check failed, and the previous image ($previous) is not pinned to a commit,"
  echo "so it cannot be rolled back to safely. Deploy a known good commit by hand:" >&2
  echo "  bash infra/azure/deploy.sh ghcr.io/md-faiz-alam/docqna-api:<good-commit-sha>" >&2
  exit 1
fi

step "Production check failed: rolling back to $previous"
deploy_to "$APP" "$previous" "${REVISION_SUFFIX}-rb"
python3 "$SMOKE_TEST" "$production_url" --quick --origin "$WEBSITE_ORIGIN" --wait 600 || true
echo "Rolled back to $previous." >&2
exit 1
