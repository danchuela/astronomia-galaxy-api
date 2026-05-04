#!/usr/bin/env python3
"""Run the pipeline in-process (no API): target → resolve + fetch → segment → measure.

From the repo root:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py M81
  python scripts/run_pipeline.py "NGC 1300"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.models import AnalyzeRequest, Target


def main() -> None:
    name = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "M51"
    request = AnalyzeRequest(
        request_id="direct-1",
        target=Target(name=name),
        task="morphology_summary",
    )

    print(f"Target: {name}")
    runner = AgentRunner(artifact_dir="artifacts", langsmith_enabled=False)
    response = runner.run(request)

    print(f"Status: {response.status}")
    print(f"Summary: {response.summary}")
    path = Path("artifacts") / request.request_id / "image.jpg"
    if path.exists():
        print(f"Image: {path}")


if __name__ == "__main__":
    main()
