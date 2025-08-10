import os
import socket
import subprocess
import sys


def check_env():
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=\n")
            f.write("TAVILY_API_KEY=\n")
            f.write("CREDIT_CARD_DATA_PATH=credit_cards_dataset.csv\n")
            f.write("FRONTEND_ORIGIN=http://localhost:5173\n")
        print("Created .env. Fill your keys & path before running.")
        # do not exit to allow first-run installation


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False


def _find_free_port(start: int, host: str = "127.0.0.1", limit: int = 20) -> int:
    p = start
    tried = 0
    while tried < limit and _port_in_use(p, host):
        p += 1
        tried += 1
    return p


def _start_frontend(api_port: int, fe_port: int):
    try:
        fe_dir = os.path.join(os.getcwd(), "frontend")
        if not os.path.isdir(fe_dir):
            return None
        # Install deps if node_modules missing
        node_modules = os.path.join(fe_dir, "node_modules")
        npm_cmd = ["npm", "install"] if not os.path.isdir(node_modules) else [
            "npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", str(fe_port)
        ]
        if npm_cmd[1] == "install":
            subprocess.call(["npm", "install"], cwd=fe_dir)
        # Start dev server with correct API base
        env = os.environ.copy()
        env["VITE_API_BASE"] = f"http://localhost:{api_port}"
        return subprocess.Popen([
            "npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", str(fe_port)
        ], cwd=fe_dir, env=env)
    except Exception:
        return None


def main():
    check_env()
    # Do not auto-install dependencies. Assume they are installed or managed externally.

    # Decide ports (avoid collisions)
    api_port = _find_free_port(8000)
    fe_port = _find_free_port(5173)

    # Propagate chosen frontend origin for CORS
    os.environ["FRONTEND_ORIGIN"] = f"http://localhost:{fe_port}"
    # Start frontend in background for local DX
    fe = _start_frontend(api_port=api_port, fe_port=fe_port)
    # start FastAPI dev server
    os.environ.setdefault("UVICORN_RELOAD", "true")
    cmd = [
        sys.executable, "-m", "uvicorn", "api.server:app",
        "--host", "0.0.0.0", "--port", str(api_port),
        "--reload"
    ]
    print(f"API running on http://localhost:{api_port}")
    print(f"Frontend running on http://localhost:{fe_port}")
    try:
        subprocess.call(cmd)
    finally:
        if fe is not None:
            try:
                fe.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
