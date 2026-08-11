# Security Policy

This repository provides model **recipes** for a **private cluster management tool**
(Fireworks) running on DGX Spark nodes. It contains no executable serving code itself;
security-sensitive behavior runs downstream in the referenced vLLM serving images and the
cluster tool.

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability. Report privately
to the project maintainers (create a private issue / security advisory through your
GitHub org, or contact the maintainers directly).

Please include:

- affected recipe / catalog entry,
- a minimal reproduction if possible,
- impact assessment.

We aim to acknowledge reports within 5 business days and to coordinate fixes with
upstream projects where relevant.

## Security notes for operators

- **Do not commit credentials.** The recipes read node addresses and credentials from
  environment variables; never bake passwords/tokens into recipes.
- **Trusted model weights only.** Weights are loaded from HF caches on the nodes; pin
  revisions and only use checkpoints you trust.
- **Network exposure:** vLLM serves on the host network (`0.0.0.0:8888`). Restrict
  access via your cluster network / firewall — do not expose the port to untrusted
  networks.
- **Image provenance:** verify image digests when loading from a registry, and follow
  upstream advisories for vLLM / PyTorch / FlashInfer / b12x.
