# Security Policy

## Supported Versions

Security fixes are evaluated against the latest `master` revision. This project is under active development and does not promise security support for old commits or redistributed executable builds.

## Reporting a Vulnerability

Please do not disclose credentials, private input data, or an exploitable vulnerability in a public issue. Use a private GitHub Security Advisory for this repository when available, or contact the repository owner through the GitHub profile and request a private channel.

Include only the information needed to reproduce the problem: affected commit, operating system, Python/dependency versions, command or input shape, impact, and a minimal reproduction. Allow time for triage before public disclosure.

## Scope

GeoFEM reads user-provided YAML and mesh/result files and can invoke configured external conversion tools. Reports involving unsafe file handling, command execution, dependency loading, credentials, or malicious input are security-relevant. Numerical instability or an incorrect FOS is normally a solver bug; report it through the standard issue form unless it also creates a security impact.
