# Security Policy

## Supported Versions
We currently provide security updates for the latest major version of KiteML.

## Reporting a Vulnerability

Please do not open a public issue. Instead, email security@kiteml.org.
We will acknowledge receipt within 48 hours and provide a timeline for the patch.

## Threat Model

KiteML handles model deserialization. Please be aware:
- **Never load untrusted `.kiteml` bundles**. Loading bundles involves unpickling, which can execute arbitrary code.
- Always use the `result.sign()` and verify cryptographic signatures when transferring models across environments.
