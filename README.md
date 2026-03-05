# Semantic Alignment in Churn Prediction Model

## CI / CD Setup

### Docker Hub Push (required secrets)

The CI workflow automatically builds and pushes the Docker image to Docker Hub on every push to `main`/`master`. To enable this, you must configure two repository secrets.

#### Steps to configure Docker Hub secrets

1. **Create a Docker Hub Personal Access Token (PAT)**
   - Log in at [https://hub.docker.com](https://hub.docker.com)
   - Go to **Account Settings → Security → New Access Token**
   - Give it a name (e.g. `github-actions`) and click **Generate**
   - Copy the token immediately — it is only shown once

   > **Important:** If your previous token was accidentally shared publicly, revoke it immediately from the same page and create a new one.

2. **Add secrets to this repository**
   - Open the repository on GitHub
   - Go to **Settings → Secrets and variables → Actions → New repository secret**
   - Add the following two secrets:

   | Secret name          | Value                              |
   |----------------------|------------------------------------|
   | `DOCKERHUB_USERNAME` | Your Docker Hub username (e.g. `barunwork04`) |
   | `DOCKERHUB_TOKEN`    | The PAT you just created (starts with `dckr_pat_`) |

   > **Note:** The secret value must be *only* the token string — do not include the `docker login` command or any other text.

3. **Verify**
   - Push a commit to `main` or `master`
   - In the **Actions** tab, the **Docker Build and Push** job will log in to Docker Hub and push the image

If the secrets are not configured, the workflow will emit a warning and skip the Docker Hub push (the GitHub Container Registry push will still proceed using the built-in `GITHUB_TOKEN`).
