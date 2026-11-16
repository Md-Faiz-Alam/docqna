# Azure infrastructure

[`main.bicep`](main.bicep) describes the live backend on **Azure Container Apps** as code:

| Setting | Value |
|---|---|
| Resource group | `docqna-rg` (its metadata lives in Central India) |
| Environment | `docqna-env` in UAE North (existing, referenced). The app's `location` must match it. |
| Container app | `docqna-api` |
| Image | `ghcr.io/md-faiz-alam/docqna-api:latest` (public, from [publish-image.yml](../../.github/workflows/publish-image.yml)) |
| Resources | 1 vCPU, 2 GiB |
| Ingress | external HTTPS, HTTP/1, target port 7860, insecure connections off |
| Scaling | 0 to 1 replicas, HTTP scale rule `http-scaler` |
| Secret | `gemini-api-key`, exposed to the container as `GEMINI_API_KEY` |
| Configuration | `CORS_ALLOWED_ORIGINS` = the Vercel site and `http://localhost:3000` |

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

## Cost

Deployments are free; Azure charges only for what runs. With minimum replicas at 0 the app costs nothing while idle, and demo traffic stays within the Consumption plan's monthly free grant. A budget with email alerts is set on the subscription.
