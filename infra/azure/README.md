# Azure infrastructure

[`main.bicep`](main.bicep) describes the live backend on **Azure Container Apps** as code:

| Setting | Value |
|---|---|
| Resource group | `docqna-rg` (its metadata lives in Central India) |
| Environment | `docqna-env` in UAE North (existing, referenced). The app's `location` must match it. |
| Container app | `docqna-api` |
| Image | `ghcr.io/md-faiz-alam/docqna-api:<commit-sha>` (public, from [publish-image.yml](../../.github/workflows/publish-image.yml)); `:latest` follows the live one |
| Revisions | Single-revision mode (Express environments allow no other); each deploy is tested on the staging app first |
| Staging app | `docqna-api-staging`: same template and settings, scales to zero, receives each new image first |
| Resources | 1 vCPU, 2 GiB |
| Ingress | external HTTPS, HTTP/1, target port 7860, insecure connections off |
| Scaling | 0 to 1 replicas, HTTP scale rule `http-scaler`, Consumption workload profile |
| Health probes | TCP startup, readiness and liveness checks on port 7860 |
| Secret | `gemini-api-key`, exposed to the container as `GEMINI_API_KEY` |
| Configuration | `CORS_ALLOWED_ORIGINS` = the Vercel site and `http://localhost:3000` |

**Verified against the live deployment:** `az deployment group what-if` reports no configuration differences between this template and the running app. The only line it shows is the read-only `runningStatus`, which a template cannot set. Earlier runs caught a region mismatch and the default health probes, both now in the template.

[`main.bicepparam`](main.bicepparam) holds the values. The Gemini key is read from the `GEMINI_API_KEY` environment variable, so it never appears in a file or on the command line.

## Deploy or check for drift

Run these in **Azure Cloud Shell** (portal toolbar → Cloud Shell → Bash). It already has the Azure CLI and Bicep, and you are signed in.

```bash
git clone https://github.com/Md-Faiz-Alam/docqna.git
cd docqna/infra/azure
read -rs GEMINI_API_KEY && export GEMINI_API_KEY   # paste the key, press Enter; nothing is shown
[ ${#GEMINI_API_KEY} -gt 20 ] && echo "key set" || echo "KEY MISSING: run the read line again"
```

Continue only after it prints `key set`. An empty key would overwrite the live secret and break the backend.

Preview what would change. Nothing is applied:

```bash
az deployment group what-if --resource-group docqna-rg --parameters main.bicepparam
```

If the preview shows only the expected changes, apply it:

```bash
az deployment group create --resource-group docqna-rg --parameters main.bicepparam
```

The deployment prints the API's URL. A changed template creates a new revision. Single-revision mode moves all traffic to it once it starts.

## Automatic deployment (staging, then production)

Every push to `main` that changes the backend goes live by itself, but only after CI passes and only after the new version has passed a full smoke test on a staging app:

```text
push to main -> CI (lint, types, tests, benchmark, Docker) -> all green?
  -> build image ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
  -> deploy to docqna-api-staging -> full smoke test
       fail: stop. Production is not touched, the run turns red
  -> deploy the same image to docqna-api -> quick check
       pass: done, :latest moved to it
       fail: previous production image redeployed (automatic rollback), the run turns red
```

The apps run in an Azure Container Apps **Express** environment, which allows only one active revision per app (`ExpressMultiRevisionModeNotSupported`), so a new version cannot be tested at 0% traffic next to the live one. A separate staging app gives the same guarantee: users never reach a version that has not passed the smoke test.

- [`ci.yml`](../../.github/workflows/ci.yml) calls [`publish-image.yml`](../../.github/workflows/publish-image.yml) after every CI job has passed. Pushes that change no backend file (docs, website, experiments) build and deploy nothing.
- [`deploy.sh`](deploy.sh) does the steps above. [`scripts/smoke_test.py`](../../scripts/smoke_test.py) checks `/health`, `/ready`, CORS for the website, a refusal in an empty session, one real cited and signed answer from the sample documents (one Gemini call), and that the per-IP rate limit holds against new session IDs and a faked `X-Forwarded-For`. It first waits until `/info` reports the new commit (`build`), so it never tests the previous version by mistake.
- GitHub signs in to Azure with **OIDC**: Azure trusts GitHub's identity for this repository's `production` environment. No Azure password or key is stored in GitHub. The identity may change only the `docqna-rg` resource group.
- The Gemini key is never touched: a deploy changes only the image.
- Within one app, Azure moves traffic to a new revision only once it has started; until then the old revision keeps serving.
- Automatic rollback needs the previous production image to be pinned to a commit. After the first deploy through the pipeline it always is; before that the app runs `:latest`, so a failed production check on that first deploy is reported for a manual fix instead.
- Until the three `AZURE_*` secrets exist, the deploy step is skipped with a notice and only the image is published.

### One-time setup

Run in **Azure Cloud Shell** (Bash).

0. Create the staging app from the same template. It reuses the Gemini key (one call per deploy) and scales to zero, so it costs nothing while idle:

```bash
git clone https://github.com/Md-Faiz-Alam/docqna.git && cd docqna/infra/azure
read -rs GEMINI_API_KEY && export GEMINI_API_KEY   # paste the key, press Enter; nothing is shown
[ ${#GEMINI_API_KEY} -gt 20 ] && echo "key set" || echo "KEY MISSING: run the read line again"
az deployment group create --resource-group docqna-rg --parameters main.bicepparam appName=docqna-api-staging image=ghcr.io/md-faiz-alam/docqna-api:latest --query properties.outputs.url.value -o tsv
```

Run the last line only after `key set`. It prints the staging address. The first pipeline run then deploys the current commit to it.

1. Create an identity for GitHub:

```bash
APP_ID=$(az ad app create --display-name docqna-github-deploy --query appId -o tsv) && az ad sp create --id $APP_ID -o none && echo $APP_ID
```

2. Let it manage only the `docqna-rg` resource group:

```bash
SUB=$(az account show --query id -o tsv) && az role assignment create --assignee $APP_ID --role Contributor --scope /subscriptions/$SUB/resourceGroups/docqna-rg -o none
```

3. Trust GitHub sign-ins from this repository's `production` environment only:

```bash
az ad app federated-credential create --id $APP_ID --parameters '{"name":"github-production","issuer":"https://token.actions.githubusercontent.com","subject":"repo:Md-Faiz-Alam@174359558/docqna@1384578467:environment:production","audiences":["api://AzureADTokenExchange"]}'
```

The subject must match what GitHub sends exactly. This repository uses GitHub's immutable subject format, which adds the owner and repository IDs (`@174359558`, `@1384578467`) so the rule keeps working after a rename. Check the prefix with `gh api repos/Md-Faiz-Alam/docqna/actions/oidc/customization/sub`. If a deploy fails with `AADSTS700213: No matching federated identity record`, the error message shows the exact subject GitHub sent; update the rule with `az ad app federated-credential update --id $APP_ID --federated-credential-id github-production --parameters '{...}'` using that subject.

4. Print the three values GitHub needs:

```bash
echo "AZURE_CLIENT_ID=$APP_ID"; echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"; echo "AZURE_SUBSCRIPTION_ID=$SUB"
```

5. On GitHub: **Settings -> Secrets and variables -> Actions -> New repository secret**, once for each of the three. They are identifiers, not passwords.

If step 1 fails with a permissions error, the account may not allow app registrations (some Azure for Students tenants block them). Deploy by hand instead (below).

Optional:

- **Approval before going live:** Settings -> Environments -> `production` -> Required reviewers. Each deploy then waits for a click.
- **Deploy only from `main`:** in the same environment, under Deployment branches, choose `main`.
- **Pull requests only:** Settings -> Branches -> add a rule for `main` that requires the CI checks to pass.

### Check a deploy

GitHub -> **Actions** -> the CI run -> **Publish image and deploy to Azure**. The run summary shows the deployed commit and the rollback command. Which image each app runs:

```bash
for app in docqna-api-staging docqna-api; do echo "$app: $(az containerapp show --name $app --resource-group docqna-rg --query 'properties.template.containers[0].image' -o tsv)"; done
```

### Roll back

Redeploy the previous image. The deploy log prints the exact command; in general:

```bash
az containerapp update --name docqna-api --resource-group docqna-rg --image ghcr.io/md-faiz-alam/docqna-api:<previous-commit-sha>
```

Azure switches traffic once the old image has started, about a minute.

### Deploy by hand

Same staging-then-production steps and smoke tests, from Cloud Shell:

```bash
git clone https://github.com/Md-Faiz-Alam/docqna.git && cd docqna && bash infra/azure/deploy.sh ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
```

The script also prints the rollback command at the end. Or re-run the pipeline: Actions -> **Publish and deploy backend** -> Run workflow.

### Bicep and the pipeline

The pipeline changes only the image; Bicep owns everything else (secret, ingress, scaling, probes, CORS) for both apps. A Bicep deploy skips the smoke test, so pass the image each app runs now:

```bash
az deployment group what-if --resource-group docqna-rg --parameters main.bicepparam image=ghcr.io/md-faiz-alam/docqna-api:<live-commit-sha>
```

For the staging app, add `appName=docqna-api-staging`.

## Cost

Deployments are free; Azure charges only for what runs. Each automatic deploy runs the staging app for a few minutes to smoke-test it, and production starts its new revision (a few hundred vCPU-seconds in total, of the monthly free grant of 180,000). The staging app scales to zero between deploys, so it costs nothing while idle. GitHub Actions minutes and GitHub Container Registry are free for a public repository. The smoke test makes one Gemini call per deploy. With minimum replicas at 0 the app costs nothing while idle, and demo traffic stays within the Consumption plan's monthly free grant. A budget with email alerts is set on the subscription.
