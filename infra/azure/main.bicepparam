using 'main.bicep'

param location = 'uaenorth'
param appName = 'docqna-api'
param environmentName = 'docqna-env'
param image = 'ghcr.io/md-faiz-alam/docqna-api:latest'
param corsAllowedOrigins = 'https://docqna-lake.vercel.app,http://localhost:3000'

// Read from the shell so the key never lands in a file or the command line.
param geminiApiKey = readEnvironmentVariable('GEMINI_API_KEY')
