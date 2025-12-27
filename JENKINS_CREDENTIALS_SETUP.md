# Jenkins Docker Hub Credentials Setup

## 🔍 Problem

Jenkins pipelines are failing with:
```
ERROR: Could not find credentials entry with ID 'dockerhub-credentials'
```

## ✅ Solution

### Option 1: Create Credentials via Jenkins UI (Recommended)

1. **Access Jenkins UI**
   - URL: http://localhost:8080 (or http://<server-ip>:8080)
   - Login with admin password: `docker exec jenkins-ci cat /var/jenkins_home/secrets/initialAdminPassword`

2. **Navigate to Credentials**
   - Click: **Manage Jenkins**
   - Click: **Credentials**
   - Click: **System**
   - Click: **Global credentials (unrestricted)**
   - Click: **Add Credentials** (on the left sidebar)

3. **Fill in Credential Details**
   - **Kind**: Username with password
   - **Scope**: Global
   - **Username**: Your Docker Hub username (e.g., `binhdang150802`)
   - **Password**: Your Docker Hub password or access token
   - **ID**: `dockerhub-credentials` (must match exactly)
   - **Description**: Docker Hub credentials for pushing images

4. **Save**
   - Click: **OK**

### Option 2: Use Docker Hub Access Token (More Secure)

Instead of using your password, create an access token:

1. **Create Access Token**
   - Go to: https://hub.docker.com/settings/security
   - Click: **New Access Token**
   - Name: `jenkins-mlops`
   - Permissions: Read & Write (or Read, Write & Delete)
   - Click: **Generate**

2. **Use Token in Jenkins**
   - Username: Your Docker Hub username
   - Password: The generated access token (not your password)

### Option 3: Skip Docker Login (Build Only, No Push)

The Jenkinsfiles have been updated to continue even if credentials are missing. However, **docker push** will fail without login.

To build images without pushing:
- The pipeline will build the image successfully
- The push step will fail (but won't stop the pipeline)
- You can manually push images later

## 🔧 Verify Credentials

After creating credentials, test by running a pipeline:

```bash
# Trigger API build in Jenkins UI
# Or check if credentials exist:
docker exec jenkins-ci cat /var/jenkins_home/credentials.xml | grep -A 5 dockerhub-credentials
```

## 📋 Quick Setup Script

Run the setup script for instructions:

```bash
chmod +x scripts/setup-jenkins-credentials.sh
./scripts/setup-jenkins-credentials.sh
```

## 🚨 Troubleshooting

### Issue: Credentials still not found after creation

**Check:**
1. Credential ID must be exactly: `dockerhub-credentials`
2. Scope must be: `Global`
3. Restart Jenkins if needed: `docker restart jenkins-ci`

### Issue: Docker push fails with "unauthorized"

**Fix:**
1. Verify credentials are correct
2. Check if Docker Hub account has access
3. Try using access token instead of password
4. Check if account has reached rate limits

### Issue: Pipeline continues but push fails

**This is expected** if credentials are missing. The pipeline will:
- ✅ Build the image successfully
- ❌ Fail at push step (but continue)
- ✅ Complete other stages

To fix: Create credentials as described above.

## 📝 All Jenkinsfiles Updated

The following Jenkinsfiles now handle missing credentials gracefully:
- ✅ `Jenkinsfile-api`
- ✅ `Jenkinsfile-streamlit`
- ✅ `Jenkinsfile-data-generator`
- ✅ `Jenkinsfile-user-simulator`

They will:
- Show a warning if credentials are missing
- Continue building the image
- Skip Docker login (push will fail, but build succeeds)

