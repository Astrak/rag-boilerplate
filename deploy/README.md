# Production deploy: secrets via AWS SSM Parameter Store

Secrets (`OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`,
`BUCKET`, `BUCKET_REGION`, `SOURCES`) live in AWS SSM Parameter Store, not in a
`.env` file on disk. `deploy/fetch-env.sh` pulls them down and regenerates
`.env` on every boot, so a server reset (lost disk, new instance, AMI
rebuild) never loses secrets — AWS is the only source of truth.

## One-time setup

1. **Store each value** (replace `<value>` — do this once per key):

   The SSM parameters themselves live in whatever region your EC2 instance
   runs in (`eu-west-3` here) — that's independent of `BUCKET_REGION`'s
   *value*, which is the S3 bucket's own region (`eu-north-1`) and is only
   ever read by the app's boto3 client, not by SSM.

   ```bash
   aws ssm put-parameter --name /rag-boilerplate/prod/OPENAI_API_KEY   --type SecureString --value '<value>' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/LANGSMITH_API_KEY --type SecureString --value '<value>' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/GOOGLE_API_KEY   --type SecureString --value '<value>' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/XAI_API_KEY      --type SecureString --value '<value>' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/BUCKET           --type SecureString --value 'rag-faiss-indext' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/BUCKET_REGION    --type SecureString --value 'eu-north-1' --region eu-west-3
   aws ssm put-parameter --name /rag-boilerplate/prod/SOURCES          --type SecureString --value '<comma-separated sources>' --region eu-west-3
   ```

2. **Attach an IAM role to the EC2 instance** (Instance Settings → Attach/Replace
   IAM role in the EC2 console, or `aws ec2 associate-iam-instance-profile`)
   with a policy scoped to just this path:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "ssm:GetParametersByPath",
         "Resource": [
           "arn:aws:ssm:eu-west-3:<account-id>:parameter/rag-boilerplate/prod",
           "arn:aws:ssm:eu-west-3:<account-id>:parameter/rag-boilerplate/prod/*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": "kms:Decrypt",
         "Resource": "arn:aws:kms:eu-west-3:<account-id>:alias/aws/ssm"
       }
     ]
   }
   ```

   Note the first `Resource` entry has no trailing `/*` — `GetParametersByPath`
   checks permissions against the path exactly as passed (no trailing
   slash/wildcard), which is a different string than the `/*` prefix pattern.
   Omitting it produces `AccessDeniedException` even though the wildcard entry
   looks like it should cover it.

   With this role attached, the instance no longer needs
   `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` anywhere — the AWS CLI and
   boto3 (used for the existing S3 knowledge-source sync) both pick up
   credentials automatically from instance metadata. `aws configure` is not
   needed on the box.

3. **Install the systemd unit** (on the EC2 instance):

   ```bash
   sudo cp /home/ec2-user/rag-boilerplate/deploy/rag-boilerplate.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now rag-boilerplate.service
   ```

4. **Retire the old bare-metal launch.** If the previous
   `nohup uvicorn ... --ssl-keyfile ... --port 443` process is still running,
   stop it (`sudo pkill -f "uvicorn apps.api.main"`) — TLS termination and
   port 443 are now handled by the `caddy` service in `docker-compose.yml`,
   and uvicorn runs as the unprivileged `api` container on its internal port
   instead.

## Day to day

- Secrets change? Update the SSM parameter, then
  `sudo systemctl restart rag-boilerplate.service` (or just
  `deploy/fetch-env.sh && docker compose up -d` by hand) to pick it up.
- `.env` on disk is now a generated artifact, not something to back up —
  it's already gitignored and gets rebuilt from SSM on every boot.
