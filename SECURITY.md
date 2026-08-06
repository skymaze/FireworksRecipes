# Security Policy

This project builds serving images for a **private cluster management tool**
(Fireworks) running on DGX Spark nodes. It does not expose a public attack surface
itself; security-sensitive behavior lives downstream in the generated vLLM serving
processes and the cluster tool.

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability. Report privately
to the project maintainers (create a private issue / security advisory through your
GitHub org, or contact the maintainers directly).

Please include:

- affected component (Dockerfile / overlay patch / recipe / scripts),
- the vLLM & dependency versions involved (`versions.conf`),
- a minimal reproduction if possible,
- impact assessment.

We aim to acknowledge reports within 5 business days and to coordinate fixes with
upstream projects where relevant.

## Security notes for operators

- **Do not commit credentials.** The scripts read node addresses and credentials
  from environment variables; never bake passwords/tokens into images or recipes.
- **Trusted model weights only.** Weights are loaded from HF caches on the nodes;
  pin revisions and only use checkpoints you trust.
- **Network exposure:** vLLM serves on the host network (`0.0.0.0:8000`). Restrict
  access via your cluster network / firewall — do not expose the port to untrusted
  networks.
- **Patch integrity:** all in-image patches are applied at build time with AST
  validation (anchors must match exactly); a failure aborts the build rather than
  silently altering code. Verify image digests when loading from a registry.
- **Keep dependencies updated:** follow upstream advisories for vLLM / PyTorch /
  FlashInfer / b12x and rebuild images when security fixes land upstream.
