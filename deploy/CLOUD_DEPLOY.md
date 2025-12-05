# Cloud Deployment Guide for AI Tutor

This guide covers deploying AI Tutor to various cloud platforms.

## Quick Start (Any Platform)

### Prerequisites
1. Docker installed locally
2. A Gemini API key from Google AI Studio
3. Cloud provider account (AWS, GCP, Azure, or others)

### Environment Variables
Always set these environment variables:
```bash
GEMINI_API_KEY=your-api-key-here
```

---

## Option 1: Deploy to Google Cloud Run (Recommended)

Cloud Run is ideal because:
- Pay only when requests are made
- Auto-scales to zero
- Easy Gemini integration (same ecosystem)

### Steps

1. **Install Google Cloud CLI**
   ```bash
   # Download from: https://cloud.google.com/sdk/docs/install
   gcloud init
   ```

2. **Enable required APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   ```

3. **Build and push the image**
   ```bash
   # From the project root
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/ai-tutor
   ```

4. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ai-tutor \
     --image gcr.io/YOUR_PROJECT/ai-tutor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="GEMINI_API_KEY=your-key" \
     --memory=512Mi \
     --cpu=1 \
     --min-instances=0 \
     --max-instances=10
   ```

5. **Get your URL**
   ```bash
   gcloud run services describe ai-tutor --format="value(status.url)"
   ```

### Cost Estimate
- ~$0-5/month for low traffic
- First 2 million requests/month free

---

## Option 2: Deploy to AWS (ECS or App Runner)

### AWS App Runner (Simpler)

1. **Push to ECR**
   ```bash
   aws ecr create-repository --repository-name ai-tutor
   
   # Login to ECR
   aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com
   
   # Build and push
   docker build -t ai-tutor -f deploy/Dockerfile .
   docker tag ai-tutor:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/ai-tutor:latest
   docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/ai-tutor:latest
   ```

2. **Create App Runner service** (via AWS Console)
   - Go to AWS App Runner
   - Create service from ECR image
   - Set environment variable: GEMINI_API_KEY
   - Choose 0.5 vCPU, 1GB RAM

### AWS ECS (More Control)

```bash
# Create cluster
aws ecs create-cluster --cluster-name ai-tutor-cluster

# Create task definition (save as task-def.json)
# Then register it
aws ecs register-task-definition --cli-input-json file://task-def.json

# Create service
aws ecs create-service \
  --cluster ai-tutor-cluster \
  --service-name ai-tutor-service \
  --task-definition ai-tutor \
  --desired-count 1 \
  --launch-type FARGATE
```

---

## Option 3: Deploy to Azure (Container Apps)

1. **Create resource group and container registry**
   ```bash
   az group create --name ai-tutor-rg --location eastus
   az acr create --resource-group ai-tutor-rg --name aitutoracr --sku Basic
   ```

2. **Build and push image**
   ```bash
   az acr build --registry aitutoracr --image ai-tutor:v1 -f deploy/Dockerfile .
   ```

3. **Create Container App**
   ```bash
   az containerapp create \
     --name ai-tutor \
     --resource-group ai-tutor-rg \
     --image aitutoracr.azurecr.io/ai-tutor:v1 \
     --target-port 8000 \
     --ingress external \
     --env-vars GEMINI_API_KEY=your-key \
     --cpu 0.5 \
     --memory 1Gi
   ```

---

## Option 4: Deploy to DigitalOcean (Budget-Friendly)

1. **Install doctl**
   ```bash
   # See: https://docs.digitalocean.com/reference/doctl/how-to/install/
   doctl auth init
   ```

2. **Create App**
   ```bash
   # Create app.yaml
   doctl apps create --spec deploy/do-app.yaml
   ```

3. **app.yaml for DigitalOcean**
   ```yaml
   name: ai-tutor
   services:
     - name: api
       dockerfile_path: deploy/Dockerfile
       source_dir: /
       http_port: 8000
       instance_count: 1
       instance_size_slug: basic-xxs
       envs:
         - key: GEMINI_API_KEY
           value: ${GEMINI_API_KEY}
           type: SECRET
   ```

---

## Option 5: Simple VPS Deployment

For a basic VPS (like $5-10/month Linode, Vultr, or Hetzner):

1. **SSH to your server**
   ```bash
   ssh user@your-server-ip
   ```

2. **Install Docker**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

3. **Clone and deploy**
   ```bash
   git clone https://github.com/yourusername/AI-Tutor.git
   cd AI-Tutor
   
   # Set API key
   export GEMINI_API_KEY=your-key-here
   
   # Start with docker-compose
   cd deploy
   docker-compose up -d
   ```

4. **Set up SSL with Caddy (optional but recommended)**
   ```bash
   # Install Caddy
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install caddy
   
   # Create Caddyfile
   echo 'yourdomain.com {
     reverse_proxy localhost:3000
   }' | sudo tee /etc/caddy/Caddyfile
   
   sudo systemctl restart caddy
   ```

---

## Scaling Recommendations

### For 1-100 users
- Single container is fine
- 0.5 CPU, 512MB RAM
- JSON file storage

### For 100-1000 users
- 2-3 containers behind load balancer
- 1 CPU, 1GB RAM each
- Switch to PostgreSQL for data

### For 1000+ users
- Auto-scaling (3-10 instances)
- Redis for session management
- PostgreSQL with read replicas
- Consider caching Gemini responses

---

## Cost Control Tips

1. **Use smaller Gemini models** for quick operations
   ```
   GEMINI_FAST_MODEL=gemini-1.5-flash  # Cheaper
   GEMINI_DEEP_MODEL=gemini-1.5-pro    # Only for complex explanations
   ```

2. **Enable auto-scaling to zero** where possible

3. **Cache common explanations** (add Redis caching layer)

4. **Monitor API usage** in Google AI Studio

---

## Security Checklist

- [ ] GEMINI_API_KEY stored as secret (not in code)
- [ ] HTTPS enabled (required for production)
- [ ] Rate limiting enabled
- [ ] CORS configured for your domain only
- [ ] Regular backups of learner data
- [ ] API key rotation every 90 days

---

## Monitoring

Add basic monitoring with these free/cheap tools:

1. **Uptime monitoring**: UptimeRobot (free)
2. **Log aggregation**: Papertrail or Logtail
3. **Error tracking**: Sentry (free tier)
4. **Performance**: Cloud provider metrics

---

## Troubleshooting

### Container won't start
```bash
docker logs ai-tutor-backend
```

### API key errors
- Verify key in Google AI Studio
- Check environment variable is set correctly
- Try key locally first

### High latency
- Check Gemini API quotas
- Consider adding response caching
- Use smaller model tier for quick responses

---

Need help? Check the [main README](../README.md) or open an issue.
