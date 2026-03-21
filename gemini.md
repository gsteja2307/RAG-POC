# Project Guidelines

This file is the canonical workspace instruction source for agent behavior in this repository.

## Secrets And Environment Files
- Do not read, search, open, summarize, quote, or inspect `.env` files or other environment-variable files unless the user explicitly asks for that.
- Treat environment files as secret-bearing inputs and avoid tools or commands that reveal their contents by default.
- If a task requires environment configuration, describe the expected variable names and purpose without reading existing env file contents.
