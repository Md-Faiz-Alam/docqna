// DocQ&A backend on Azure Container Apps.
//
// Describes the live container app: a public image from GitHub Container
// Registry, 1 vCPU / 2 GiB, HTTPS ingress on port 7860, and HTTP scaling
// between 0 and 1 replicas so it costs nothing while idle. The same template
// creates the staging app (appName=docqna-api-staging); see README.md.
//
// The Container Apps environment already exists and is referenced, not
// created. Deploy steps: infra/azure/README.md.

// The app must live in its environment's region. The resource group itself is
// in Central India, so resourceGroup().location cannot be used here.
@description('Azure region of the Container Apps environment.')
param location string = 'uaenorth'

@description('Name of the container app.')
param appName string = 'docqna-api'

@description('Name of the existing Container Apps environment.')
param environmentName string = 'docqna-env'

@description('Backend image, built by .github/workflows/publish-image.yml. Pass the live commit SHA tag (see README).')
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
    workloadProfileName: 'Consumption'
    configuration: {
      // Express environments allow one active revision per app. New versions
      // are tested on a separate staging app instead (infra/azure/deploy.sh).
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'http'
        allowInsecure: false
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
          // TCP health checks on the app port, matching the ones Azure adds by
          // default. Startup allows up to 4 minutes for the models to load.
          probes: [
            {
              type: 'Liveness'
              tcpSocket: {
                port: targetPort
              }
              initialDelaySeconds: 0
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
              successThreshold: 1
            }
            {
              type: 'Readiness'
              tcpSocket: {
                port: targetPort
              }
              initialDelaySeconds: 0
              periodSeconds: 5
              timeoutSeconds: 5
              failureThreshold: 48
              successThreshold: 1
            }
            {
              type: 'Startup'
              tcpSocket: {
                port: targetPort
              }
              initialDelaySeconds: 1
              periodSeconds: 1
              timeoutSeconds: 3
              failureThreshold: 240
              successThreshold: 1
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
