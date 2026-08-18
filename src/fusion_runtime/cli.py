from __future__ import annotations

import argparse
import json

import uvicorn

from .config import load_spec
from .gateway import create_app
from .runtime import FusionRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a FusionSpec recipe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the recipe, print a secret-free summary, and exit",
    )
    args = parser.parse_args()
    spec = load_spec(args.config)
    if args.check:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "public_model": spec.serve.model_name,
                    "providers": sorted(spec.providers),
                    "models": sorted(spec.models),
                    "pools": sorted(spec.pools),
                    "policy": spec.policy.type,
                    "protocols": sorted(spec.serve.protocols),
                },
                indent=2,
            )
        )
        return
    uvicorn.run(create_app(FusionRuntime(spec)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
