param location string
param storageAccountName string

resource storageAccount 'Microsoft.Storage/storageAccounts@2026-04-01' = {
    name: storageAccountName
    location: location
    sku: {
        name: 'Standard_LRS'
    }
    kind: 'StorageV2'
    properties: {
        accessTier: 'Hot'
    }
}

output storageAccountId string = storageAccount.id
