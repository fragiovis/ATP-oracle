#!/bin/bash
# Bootstrap script for EC2 – runs atp-features + atp-train on S3 data.
# Designed for Amazon Linux 2023 (ami-0c101f26f147fa7fd, us-east-1).
# Run as user-data or SSH into the instance and execute manually.

set -euo pipefail
exec > >(tee /var/log/atp-oracle-bootstrap.log) 2>&1

echo "=== ATP Oracle EC2 Bootstrap ==="
echo "Started: $(date)"

# ── System packages ────────────────────────────────────────────────────
dnf update -y -q
dnf install -y -q git python3.12 python3.12-pip python3.12-devel gcc

# ── Clone project ──────────────────────────────────────────────────────
cd /opt
git clone https://github.com/fragiovis/project1_atp_oracle.git atp-oracle 2>/dev/null \
  || (cd atp-oracle && git pull)
cd atp-oracle

# ── Python virtual env ─────────────────────────────────────────────────
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q
pip install boto3 shap -q

# ── AWS region (bucket is us-east-1) ──────────────────────────────────
export AWS_DEFAULT_REGION=us-east-1

# ── Run pipeline ───────────────────────────────────────────────────────
echo ""
echo "=== Step 1: atp-features ==="
atp-features
echo "Features done: $(date)"

echo ""
echo "=== Step 2: atp-train ==="
atp-train
echo "Training done: $(date)"

echo ""
echo "=== All done! Results written to S3. ==="
echo "Finished: $(date)"

# Auto-terminate the instance when done (remove this line if you want to SSH in)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1
