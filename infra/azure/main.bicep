// DocQ&A backend on Azure Container Apps.
//
// Describes the live container app: a public image from GitHub Container
// Registry, 1 vCPU / 2 GiB, HTTPS ingress on port 7860, and HTTP scaling
// between 0 and 1 replicas so it costs nothing while idle.
//
// The Container Apps environment already exists and is referenced, not
// created. Deploy steps: infra/azure/README.md.

@description('Azure region. Must match the environment.')
param location string = resourceGroup().location

@description('Name of the container app.')
param appName string = 'docqna-api'

@description('Name of the existing Container Apps environment.')
param environmentName string = 'docqna-env'

@description('Backend image, built by .github/workflows/publish-image.yml.')
param image string = 'ghcr.io/md-faiz-alam/docqna-api:latest'

@description('Website origins allowed to call the API, comma-separated.')
param corsAllowedOrigins string = 'https://docqna-lake.vercel.app,http://localhost:3000'

@description('Google Gemini API key. Stored as a Container Apps secret.')
@secure()
param geminiApiKey string

@description('Port the container listens on (the image default).')
param targetPort int = 7860

@description('Concurrent requests per replica before scaling out.')
param concurrentRequests int = 10

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'http'
        allowInsecure: false
        clientCertificateMode: 'ignore'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      secrets: [
        {
          name: 'gemini-api-key'
          value: geminiApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: appName
          image: image
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'CORS_ALLOWED_ORIGINS'
              value: corsAllowedOrigins
            }
            {
              name: 'GEMINI_API_KEY'
              secretRef: 'gemini-api-key'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http-scaler'
            http: {
              metadata: {
                concurrentRequests: string(concurrentRequests)
              }
            }
          }
        ]
      }
    }
  }
}

@description('Public HTTPS address of the API.')
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
