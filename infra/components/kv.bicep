param location string
param environmentName string

var vaultName = 'kv${toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))}'

resource vault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  name: vaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    publicNetworkAccess: 'Enabled'
  }
}


output name string = vault.name
output id string = vault.id
output uri string = vault.properties.vaultUri
