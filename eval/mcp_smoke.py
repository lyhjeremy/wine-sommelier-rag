"""Start the MCP server over stdio, list its tools and call both.

    python eval/mcp_smoke.py        # writes eval/mcp_smoke.json
"""
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
CALLS = [
    ("parse_constraints", {"request": "a Pinot Noir from Oregon under twenty bucks"}),
    ("search_wines", {"query": "earthy Pinot Noir with cherry", "provinces": ["Oregon"],
                      "varieties": ["Pinot Noir"], "max_price": 25, "k": 3}),
    ("search_wines", {"query": "crisp white", "countries": ["Narnia"]}),
]


async def main():
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(ROOT / "src" / "mcp_server.py")], cwd=str(ROOT))
    t0 = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            out = {"date": date.today().isoformat(), "server": init.server_info.name,
                   "startup_s": round(time.perf_counter() - t0, 2),
                   "tools": [t.name for t in listed.tools], "calls": []}
            for name, args in CALLS:
                t1 = time.perf_counter()
                res = await session.call_tool(name, args)
                text = "".join(getattr(c, "text", "") for c in res.content)
                out["calls"].append({"tool": name, "args": args, "is_error": res.is_error,
                                     "seconds": round(time.perf_counter() - t1, 2),
                                     "result": json.loads(text) if text.startswith("{") else text})
    return out


if __name__ == "__main__":
    r = asyncio.run(main())
    (ROOT / "eval" / "mcp_smoke.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    for c in r["calls"]:
        print(c["tool"], c["seconds"], json.dumps(c["result"], ensure_ascii=False)[:300])
