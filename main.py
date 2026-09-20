import uvicorn

from routes import app

# mcp_key = os.getenv("OBSIDIAN_MCP_KEY")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
