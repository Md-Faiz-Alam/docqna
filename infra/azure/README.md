# Azure infrastructure

[`main.bicep`](main.bicep) describes the live backend on **Azure Container Apps** as code:

| Setting | Value |
|---|---|
| Resource group | `docqna-rg` (its metadata lives in Central India) |
| Environment | `docqna-env` in UAE North (existing, referenced). The app's `location` must match it. |
| Container app | `docqna-api` |
| Image | `ghcr.io/md-faiz-alam/docqna-api:<commit-sha>` (public, from [publish-image.yml](../../.github/workflows/publish-image.yml)); `:latest` follows the live one |
| Revisions | Multiple-revision mode: each deploy is a new revision, smoke-tested at 0% traffic before it gets 100% |
| Resources | 1 vCPU, 2 GiB |
| Ingress | external HTTPS, HTTP/1, target port 7860, insecure connections off |
| Scaling | 0 to 1 replicas, HTTP scale rule `http-scaler`, Consumption workload profile |
| Health probes | TCP startup, readiness and liveness checks on port 7860 |
| Secret | `gemini-api-key`, exposed to the container as `GEMINI_API_KEY` |
| Configuration | `CORS_ALLOWED_ORIGINS` = the Vercel site and `http://localhost:3000` |

**Verified against the live deployment** (before the switch to multiple-revision mode; until the pipeline's first deploy, `what-if` also shows `activeRevisionsMode` changing from `Single`): `az deployment group what-if` reports no configuration differences between this template and the running app. The only line it shows is the read-only `runningStatus`, which a template cannot set. Earlier runs caught a region mismatch and the default health probes, both now in the template.

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

## Automatic deployment (blue/green)

Every push to `main` that changes the backend goes live by itself, but only after CI passes and only if the new version passes a smoke test first:

```text
push to main -> CI (lint, types, tests, benchmark, Docker) -> all green?
  -> build image ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
  -> new revision at 0% traffic (users stay on the old one)
  -> smoke test at the new revision's own address
       pass: 100% of traffic to the new revision, :latest moved to it
       fail: new revision deactivated, nothing changes for users, the run turns red
```

- [`ci.yml`](../../.github/workflows/ci.yml) calls [`publish-image.yml`](../../.github/workflows/publish-image.yml) after every CI job has passed. Pushes that change no backend file (docs, website, experiments) build and deploy nothing.
- [`deploy.sh`](deploy.sh) does the blue/green steps. [`scripts/smoke_test.py`](../../scripts/smoke_test.py) checks `/health`, `/ready`, CORS for the website, a refusal in an empty session, one real cited and signed answer from the sample documents (one Gemini call), and that the per-IP rate limit holds against new session IDs and a faked `X-Forwarded-For`.
- GitHub signs in to Azure with **OIDC**: Azure trusts GitHub's identity for this repository's `production` environment. No Azure password or key is stored in GitHub. The identity may change only the `docqna-rg` resource group.
- The Gemini key is never touched: a deploy changes only the image.
- The previous revision stays active at 0% traffic, so a rollback is one traffic switch. Older revisions are deactivated.
- Until the three `AZURE_*` secrets exist, the deploy step is skipped with a notice and only the image is published.

### One-time setup

Run in **Azure Cloud Shell** (Bash).

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
az ad app federated-credential create --id $APP_ID --parameters '{"name":"github-production","issuer":"https://token.actions.githubusercontent.com","subject":"repo:Md-Faiz-Alam/docqna:environment:production","audiences":["api://AzureADTokenExchange"]}'
```

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

GitHub -> **Actions** -> the CI run -> **Publish image and deploy to Azure**. The run summary shows the deployed commit and the rollback command. In Cloud Shell:

```bash
az containerapp revision list --name docqna-api --resource-group docqna-rg --query "[].{name:name, active:properties.active, traffic:properties.trafficWeight, health:properties.healthState, created:properties.createdTime}" -o table
```

### Roll back

Send all traffic back to the previous revision (still active at 0%, so this is instant):

```bash
az containerapp ingress traffic set --name docqna-api --resource-group docqna-rg --revision-weight <previous-revision>=100 <current-revision>=0
```

To go back further, deploy an older image by hand (below) with its commit SHA.

### Deploy by hand

Same blue/green steps and smoke test, from Cloud Shell:

```bash
git clone https://github.com/Md-Faiz-Alam/docqna.git && cd docqna && bash infra/azure/deploy.sh ghcr.io/md-faiz-alam/docqna-api:<commit-sha>
```

The script also prints the rollback command at the end. Or re-run the pipeline: Actions -> **Publish and deploy backend** -> Run workflow.

### Bicep and the pipeline

The pipeline changes the image and the revision traffic; Bicep owns everything else (secret, ingress, scaling, probes, CORS). A Bicep deploy skips the smoke test and sends traffic to the newest revision, so pass the image that is live now:

```bash
az deployment group what-if --resource-group docqna-rg --parameters main.bicepparam image=ghcr.io/md-faiz-alam/docqna-api:<live-commit-sha>
```

## Cost

Deployments are free; Azure charges only for what runs. Each automatic deploy starts the new revision for a few minutes to smoke-test it (a few hundred vCPU-seconds of the monthly free grant of 180,000), and the old revision kept for rollback scales to zero like any other. GitHub Actions minutes and GitHub Container Registry are free for a public repository. The smoke test makes one Gemini call per deploy. With minimum replicas at 0 the app costs nothing while idle, and demo traffic stays within the Consumption plan's monthly free grant. A budget with email alerts is set on the subscription.
