# Deploy API to Kubernetes

Follow these steps to deploy the OriginHub Agentic API to your existing Kubernetes cluster.

## Step 1: Build and Push Docker Image

```bash
# Build the Docker image
docker build -f k8s/Dockerfile -t originhub/agentic-api:latest .

# Tag for your registry (replace with your registry)
docker tag originhub/agentic-api:latest your-registry/originhub/agentic-api:latest

# Push to registry
docker push your-registry/originhub/agentic-api:latest
```

### Common Registries:

**Docker Hub:**

```bash
docker tag originhub/agentic-api:latest username/originhub-agentic-api:latest
docker push username/originhub-agentic-api:latest
```

**Google Container Registry (GCR):**

```bash
docker tag originhub/agentic-api:latest gcr.io/YOUR_PROJECT_ID/agentic-api:latest
docker push gcr.io/YOUR_PROJECT_ID/agentic-api:latest
```

**AWS ECR:**

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag originhub/agentic-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/agentic-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/agentic-api:latest
```

## Step 2: Update Deployment Manifest

Edit `k8s/api-deployment.yaml` and update the image:

```yaml
spec:
  template:
    spec:
      containers:
        - name: agentic-api
          image: your-registry/originhub/agentic-api:latest # Update this
```

### Update API Key (if using different key):

```yaml
stringData:
  OPENAI_API_KEY: "your-actual-api-key-here"
```

### Update Weaviate Endpoints:

```yaml
data:
  WEAVIATE_HOST: "136.111.90.112"
  WEAVIATE_PORT: "80"
  WEAVIATE_GRPC_HOST: "34.44.210.201"
  WEAVIATE_GRPC_PORT: "50051"
```

## Step 3: Deploy to Kubernetes

```bash
# Deploy
kubectl apply -f k8s/api-deployment.yaml

# Check deployment status
kubectl get pods -n originhub-agentic

# Expected output:
# NAME                          READY   STATUS    RESTARTS   AGE
# agentic-api-xxx-yyy           1/1     Running   0          2m
# agentic-api-xxx-zzz           1/1     Running   0          2m
```

## Step 4: Verify Deployment

```bash
# Check service
kubectl get svc -n originhub-agentic

# Expected output:
# NAME          TYPE           CLUSTER-IP    EXTERNAL-IP    PORT(S)
# agentic-api   LoadBalancer   10.0.0.1      34.101.x.x     8000:31234/TCP

# Get external IP (may take a few minutes)
EXTERNAL_IP=$(kubectl get svc agentic-api -n originhub-agentic -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test the API
curl http://$EXTERNAL_IP:8000/health
```

## Step 5: View Logs

```bash
# View logs from all pods
kubectl logs -f deployment/agentic-api -n originhub-agentic

# View logs from specific pod
kubectl logs pod/agentic-api-xxx-yyy -n originhub-agentic
```

## Step 6: Port Forward (for testing)

```bash
# Forward local port 8000 to API service
kubectl port-forward svc/agentic-api 8000:8000 -n originhub-agentic

# Test
curl http://localhost:8000/health
```

## Troubleshooting

### Pod in CrashLoopBackOff

```bash
# Check logs
kubectl logs pod/agentic-api-xxx-yyy -n originhub-agentic --previous

# Check pod details
kubectl describe pod agentic-api-xxx-yyy -n originhub-agentic
```

### Image not found error

Make sure to:

1. Push image to registry
2. Update image URL in deployment.yaml
3. If using private registry, create image pull secret:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=your-registry \
  --docker-username=username \
  --docker-password=password \
  -n originhub-agentic
```

Then add to deployment.yaml:

```yaml
spec:
  template:
    spec:
      imagePullSecrets:
        - name: regcred
```

### Health check failing

```bash
# Port forward and test manually
kubectl port-forward pod/agentic-api-xxx-yyy 8000:8000 -n originhub-agentic

# In another terminal
curl http://localhost:8000/health

# Check logs for errors
kubectl logs pod/agentic-api-xxx-yyy -n originhub-agentic
```

## Update Deployment

```bash
# Update image
kubectl set image deployment/agentic-api \
  agentic-api=your-registry/originhub/agentic-api:v1.1.0 \
  -n originhub-agentic

# Monitor rollout
kubectl rollout status deployment/agentic-api -n originhub-agentic

# Rollback if needed
kubectl rollout undo deployment/agentic-api -n originhub-agentic
```

## Scale Deployment

```bash
# Scale to 5 replicas
kubectl scale deployment agentic-api --replicas=5 -n originhub-agentic

# Check scaling
kubectl get pods -n originhub-agentic
```

## Delete Deployment

```bash
# Delete all resources
kubectl delete -f k8s/api-deployment.yaml

# Or delete specific resources
kubectl delete deployment agentic-api -n originhub-agentic
kubectl delete svc agentic-api -n originhub-agentic
kubectl delete namespace originhub-agentic
```

## Useful Commands

```bash
# Get all resources
kubectl get all -n originhub-agentic

# Describe deployment
kubectl describe deployment agentic-api -n originhub-agentic

# Get resource usage
kubectl top pods -n originhub-agentic

# Edit deployment
kubectl edit deployment agentic-api -n originhub-agentic

# Port forward with specific pod
kubectl port-forward pod/agentic-api-xxx-yyy 8000:8000 -n originhub-agentic

# Execute command in pod
kubectl exec -it pod/agentic-api-xxx-yyy -n originhub-agentic -- /bin/bash

# Check events
kubectl get events -n originhub-agentic
```

---

**Need help?** Run: `kubectl get pods -n originhub-agentic` to see your pods' status
