from __future__ import annotations

import re

from pydantic import BaseModel, Field

_AGENT_HEADER = re.compile(r"^Agent (\d+)\s*$", re.MULTILINE)
_TOP_LEVEL_FIELD = re.compile(r"^  ([^\s:][^:]*):\s*(.*)$")


class GpuAgent(BaseModel):

    agent_index: int = Field(gt=0)
    marketing_name: str
    gfx_target: str
    compute_units: int = Field(gt=0)
    max_clock_mhz: int | None = None


def _top_level_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = _TOP_LEVEL_FIELD.match(line)
        if match:
            fields.setdefault(match.group(1).strip(), match.group(2).strip())
    return fields


def _need(fields: dict[str, str], key: str, agent: int) -> str:
    value = fields.get(key)
    if not value:
        raise ValueError(f"rocminfo agent {agent} has no '{key}' line")
    return value


def parse_rocminfo(text: str) -> list[GpuAgent]:
  
    headers = list(_AGENT_HEADER.finditer(text))
    if not headers:
        raise ValueError("rocminfo output has no 'Agent N' sections")
    gpus: list[GpuAgent] = []
    for position, header in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        agent = int(header.group(1))
        fields = _top_level_fields(text[header.end() : end])
        if fields.get("Device Type") != "GPU":
            continue
        target = _need(fields, "Name", agent)
        if not target.startswith("gfx"):
            raise ValueError(f"rocminfo agent {agent} has an unexpected GPU name '{target}'")
        clock = fields.get("Max Clock Freq. (MHz)")
        gpus.append(
            GpuAgent(
                agent_index=agent,
                marketing_name=_need(fields, "Marketing Name", agent),
                gfx_target=target,
                compute_units=int(_need(fields, "Compute Unit", agent)),
                max_clock_mhz=int(clock) if clock else None,
            )
        )
    return gpus
