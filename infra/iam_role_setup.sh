#!/bin/bash
# Creates the IAM role + instance profile needed by the EC2 bootstrap.
# Run this ONCE from your local machine before launching EC2.

set -euo pipefail

ROLE_NAME="atp-oracle-s3-role"
BUCKET="tennis-predictor-959366206165-us-east-1-an"

echo "Creating IAM role: $ROLE_NAME"

# Trust policy: allows EC2 to assume the role
aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Principal":{"Service":"ec2.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }' \
  --description "Allows EC2 to read/write the ATP Oracle S3 bucket" 2>/dev/null \
  || echo "Role already exists – skipping."

# Inline policy: S3 access to the specific bucket only
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "s3-atp-oracle-bucket" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{
      \"Effect\":\"Allow\",
      \"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],
      \"Resource\":[
        \"arn:aws:s3:::${BUCKET}\",
        \"arn:aws:s3:::${BUCKET}/*\"
      ]
    },{
      \"Effect\":\"Allow\",
      \"Action\":\"ec2:TerminateInstances\",
      \"Resource\":\"*\"
    }]
  }"

# Instance profile (wraps the role for EC2)
aws iam create-instance-profile \
  --instance-profile-name "$ROLE_NAME" 2>/dev/null \
  || echo "Instance profile already exists – skipping."

aws iam add-role-to-instance-profile \
  --instance-profile-name "$ROLE_NAME" \
  --role-name "$ROLE_NAME" 2>/dev/null \
  || echo "Role already attached – skipping."

echo ""
echo "Done. IAM role '$ROLE_NAME' is ready."
echo "Now launch the EC2 instance with --iam-instance-profile Name=$ROLE_NAME"
