param location string
param environmentName string
param containerAppEnvironmentId string
param registryName string
param registryLoginServer string
@description('Public HTTPS URL of the separately deployed API Container App.')
param apiUri string
@description('Client ID used by the Flutter web SPA. This value is public.')
param entraApplicationClientId string
@description('Fully qualified image reference hosted in the project Azure Container Registry.')
param imageName string

var serviceName = 'web'
var suffix = toLower(uniqueString(resourceGroup().id))
// Container App names are limited to 32 characters. The previous descriptive
// prefix plus the unique suffix exceeded that limit in this resource group.
var appName = 'mr-web-${suffix}'
var registryIdentityName = 'id-${serviceName}-${suffix}'

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource registryIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: registryIdentityName
  location: location
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, registryIdentity.id, 'AcrPull')
  scope: registry
  properties: {
    principalId: registryIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  dependsOn: [
    acrPull
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${registryIdentity.id}': {}
    }
  }
  tags: {
    'azd-service-name': serviceName
    'azd-env-name': environmentName
  }
  properties: {
    managedEnvironmentId: containerAppEnvironmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: registryLoginServer
          identity: registryIdentity.id
        }
      ]
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: serviceName
          image: imageName
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            {
              name: 'API_BASE_URL'
              value: apiUri
            }
            {
              name: 'ENTRA_CLIENT_ID'
              value: entraApplicationClientId
            }
            {
              name: 'ENTRA_TENANT_ID'
              value: tenant().tenantId
            }
            {
              name: 'API_SCOPE'
              value: 'api://${entraApplicationClientId}/user_impersonation'
            }
          ]
        }
      ]
    }
  }
}

output id string = app.id
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
