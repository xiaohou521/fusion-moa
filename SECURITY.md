# Security policy

Do not report credentials, private endpoints, prompt contents, or an exploitable
deployment issue in a public ticket. Once the repository is hosted on GitHub,
use its private security-advisory flow. Until then, contact the repository owner
through the private channel by which you received access.

The supported line is the latest `0.1.x` release. Reports should include the
version, affected protocol/provider/policy, minimal reproduction with secrets
removed, impact, and whether the issue crosses the expert/main trust boundary.

The runtime treats recipes and installed Python/DeepSeek Harness plugins as
trusted deployment code. Model output, user input, and expert advice are
untrusted data. Environment variables referenced by recipes are secrets and
must never appear in logs, traces, response headers, performance cards, or test
fixtures.
