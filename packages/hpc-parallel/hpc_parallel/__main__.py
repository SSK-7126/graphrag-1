# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Allow running hpc_parallel via `python -m hpc_parallel`."""

from __future__ import annotations

import sys

from hpc_parallel.cli import main

if __name__ == "__main__":
    sys.exit(main())
