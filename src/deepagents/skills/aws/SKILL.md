---
name: aws
description: AWS service selection, boto3, AWS CLI, architecture, security, and cost guidance.
---

# AWS Skill

Use for AWS services, deployment, boto3, AWS CLI, IAM, architecture, security, or cost.

## Rules

- Recommend the simplest managed service that fits; explain the tradeoff.
- Use least-privilege IAM. Never suggest wildcard actions and resources together.
- Never hard-code credentials. Prefer roles, profiles, or environment configuration.
- Enable encryption and block public access by default.
- Include an explicit region, runnable code, and relevant pricing model.
- Prefer IaC for repeatable infrastructure.
