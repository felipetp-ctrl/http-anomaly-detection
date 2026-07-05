#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${AZURE_CONTAINER_APP_NAME:?Set AZURE_CONTAINER_APP_NAME}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
REGISTRY="${AZURE_CONTAINER_REGISTRY:?Set AZURE_CONTAINER_REGISTRY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CANARY_PERCENT="${CANARY_PERCENT:-10}"

IMAGE="${REGISTRY}.azurecr.io/http-anomaly-detection:${IMAGE_TAG}"

echo "=== Building and pushing image ==="
az acr build \
  --registry "$REGISTRY" \
  --image "http-anomaly-detection:${IMAGE_TAG}" \
  .

echo "=== Deploying canary revision (${CANARY_PERCENT}% traffic) ==="
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$IMAGE" \
  --revision-suffix "v${IMAGE_TAG}" \
  --set-env-vars \
    "AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn-string" \
    "FEATURE_LOG_CONTAINER=feature-logs"

NEW_REVISION=$(az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[-1].name" -o tsv)

PROD_REVISION=$(az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[-2].name" -o tsv)

az containerapp ingress traffic set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --revision-weight \
    "${PROD_REVISION}=$((100 - CANARY_PERCENT))" \
    "${NEW_REVISION}=${CANARY_PERCENT}"

echo "=== Canary deployed ==="
echo "Production: ${PROD_REVISION} ($((100 - CANARY_PERCENT))%)"
echo "Canary: ${NEW_REVISION} (${CANARY_PERCENT}%)"
echo ""
echo "To promote canary to 100%:"
echo "  az containerapp ingress traffic set \\"
echo "    --name $APP_NAME --resource-group $RESOURCE_GROUP \\"
echo "    --revision-weight ${NEW_REVISION}=100"
