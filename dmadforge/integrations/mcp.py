"""MCP (Model Context Protocol) server entrypoint (optional).

Run with:  python -m dmadforge.integrations.mcp
Requires the `mcp` package; gracefully degrades with a clear message.
"""


def main():
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
    except Exception:
        print("mcp package not installed; install with: pip install mcp", flush=True)
        return

    from ..config import load_config
    from ..model.engine import DmadEngine

    engine = DmadEngine(load_config())
    server = Server("dmadforge")

    @server.tool()
    def generate(prompt: str, n_frames: int = 4, seed: str = "mcp") -> dict:
        return engine.generate(prompt, n_frames=n_frames, seed=seed)

    import asyncio
    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write)

    asyncio.run(run())


if __name__ == "__main__":
    main()
