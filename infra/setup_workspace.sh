#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
WORKSPACE="${AZURE_ML_WORKSPACE:?Set AZURE_ML_WORKSPACE}"
LOCATION="${AZURE_LOCATION:-eastus}"
COMPUTE_NAME="${AZURE_ML_COMPUTE:-cpu-cluster}"
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-httpanomalystorage}"

echo "=== Creating Resource Group ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== Creating Azure ML Workspace ==="
az ml workspace create \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "=== Creating Compute Cluster ==="
az ml compute create \
  --name "$COMPUTE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --type AmlCompute \
  --size Standard_DS2_v2 \
  --min-instances 0 \
  --max-instances 2

echo "=== Creating Storage Account for Feature Logs ==="
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS

az storage container create \
  --name feature-logs \
  --account-name "$STORAGE_ACCOUNT"

echo "=== Done ==="
echo "Workspace: $WORKSPACE"
echo "Compute: $COMPUTE_NAME"
echo "Storage: $STORAGE_ACCOUNT"
