# Acme Cloud Security and Compliance

Acme Cloud is designed to meet the security needs of regulated industries.

## Encryption

All customer data is encrypted at rest using AES-256 and in transit using
TLS 1.3. Encryption keys are managed by the Acme Key Management Service and
rotated automatically every 90 days. Enterprise customers may bring their own
keys (BYOK) via the AWS KMS integration.

## Compliance Certifications

Acme Cloud maintains SOC 2 Type II, ISO 27001, and GDPR compliance. A HIPAA
Business Associate Agreement (BAA) is available to Enterprise customers on
request. Audit reports can be downloaded from the Trust Center once an NDA is
signed.

## Authentication

The platform supports multi-factor authentication (MFA) for all tiers. Single
sign-on (SSO) via SAML 2.0 and SCIM user provisioning are available on the
Enterprise tier only. Passwords must be at least 12 characters and are checked
against a breached-password list.

## Data Residency

Customer data can be pinned to one of three regions: US (us-east-1), EU
(eu-west-1), or APAC (ap-southeast-2). Data residency is configured at account
creation and cannot be changed afterward without a data migration request.

## Incident Response

Acme maintains a 24/7 security operations centre. Confirmed security incidents
affecting customer data are disclosed to affected customers within 72 hours, in
line with GDPR breach-notification requirements.
