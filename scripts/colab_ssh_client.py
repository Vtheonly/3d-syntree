#!/usr/bin/env python3
"""colab_ssh_client.py — the agent-sandbox side of the OPS-01 SSH bridge (#36).

Connects to an authorized Colab/Kaggle Tesla T4 runtime through a public
TCP tunnel endpoint (the runtime side is prepared by
``notebooks/run_3d_syntree.ipynb`` Cell 3).  Pure-Python (paramiko) so it
runs in sandboxes without an OpenSSH binary.

Authorization boundary (constitution #45 §6/§28.2/§29.1): the operator
starts the runtime + tunnel with their own account and hands the agent
the endpoint; this client only then connects, with the sandbox keypair
(private key never leaves the sandbox secrets dir, never committed).

Subcommands
-----------
verify   Run the §29.2 hardware preflight battery + repo state and save a
         JSON transcript (evidence for #36/#43).
exec     Run an arbitrary command on the runtime, stream stdout/stderr.
put/get  SFTP file transfer (get supports recursive directory fetch).
tail     Poll a remote file for N seconds and print appended lines.

All commands print a machine-readable exit status; failures are loud,
never silent.  No secrets are logged or embedded.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover - loud, actionable failure
    sys.stderr.write(
        "paramiko is required: pip install paramiko\n"
    )
    raise

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_KEY = os.environ.get(
    "SYNTREE_SSH_KEY", "/home/z/my-project/.secrets/colab_ed25519"
)
DEFAULT_USER = os.environ.get("SYNTREE_SSH_USER", "root")
EVIDENCE_DIR = Path(
    os.environ.get("SYNTREE_SSH_EVIDENCE", "/home/z/my-project/ssh_evidence")
)


def _fail(msg: str, code: int = 2) -> "None":
    sys.stderr.write(f"[colab_ssh_client] FAIL: {msg}\n")
    sys.exit(code)


def connect(
    host: str,
    port: int,
    user: str = DEFAULT_USER,
    key_path: str = DEFAULT_KEY,
    timeout: float = 30.0,
    banner_wait: float = 90.0,
    proxy_command: str | None = None,
) -> "paramiko.SSHClient":
    """Open the authorized SSH session.  Fails loudly on any error.

    ``proxy_command`` (e.g. ``colab ssh --proxy-mode -s NAME -i KEY``)
    routes the transport through an OpenSSH-compatible stdio bridge
    instead of a direct host:port socket.
    """
    if not Path(key_path).is_file():
        _fail(f"private key not found: {key_path} (set SYNTREE_SSH_KEY)")
    client = paramiko.SSHClient()
    # The tunnel endpoint identity changes per session; host keys are
    # pinned per-connection by the operator-provided endpoint itself.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = None
    if proxy_command:
        sock = paramiko.ProxyCommand(proxy_command)
        try:
            client.connect(
                hostname=host or "colab-runtime",
                username=user,
                key_filename=key_path,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
                banner_timeout=timeout,
                auth_timeout=timeout,
                sock=sock,
            )
        except paramiko.ssh_exception.AuthenticationException as exc:
            _fail(
                "authentication failed through proxy — is the sandbox public "
                f"key the one passed to the bridge? ({exc})",
                code=3,
            )
        except Exception as exc:  # noqa: BLE001
            _fail(f"could not connect through proxy: {exc}", code=4)
    else:
        try:
            # Tunnels can take ~90s to accept the first connection (ngrok
            # cold start); retry within the banner wait window.
            deadline = time.time() + banner_wait
            while True:
                try:
                    client.connect(
                        hostname=host,
                        port=port,
                        username=user,
                        key_filename=key_path,
                        timeout=timeout,
                        allow_agent=False,
                        look_for_keys=False,
                        banner_timeout=timeout,
                        auth_timeout=timeout,
                    )
                    break
                except (paramiko.ssh_exception.SSHException, socket.error) as exc:
                    if time.time() >= deadline:
                        raise
                    sys.stderr.write(
                        f"[colab_ssh_client] connect retry ({exc}); retrying...\n"
                    )
                    time.sleep(5.0)
        except paramiko.ssh_exception.AuthenticationException as exc:
            _fail(
                "authentication failed — is the sandbox public key installed in "
                f"the runtime's authorized_keys (notebook Cell 3)? ({exc})",
                code=3,
            )
        except Exception as exc:  # noqa: BLE001 - report exactly what failed
            _fail(f"could not connect to {host}:{port}: {exc}", code=4)
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(15.0)
    return client


def run(
    client: "paramiko.SSHClient", cmd: str, timeout: float = 600.0
) -> tuple[int, str]:
    """Execute ``cmd``; return (exit_status, combined output)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    text = out + (("\n[stderr]\n" + err) if err.strip() else "")
    return status, text


# ---------------------------------------------------------------------------
# verify — the §29.2 hardware preflight battery
# ---------------------------------------------------------------------------

VERIFY_CMDS: list[tuple[str, str]] = [
    ("whoami", "whoami"),
    ("date_utc", "date -u +%Y-%m-%dT%H:%M:%SZ"),
    ("hostname", "hostname"),
    ("repo_commit", "git -C /content/3d-syntree rev-parse HEAD"),
    ("repo_branch", "git -C /content/3d-syntree rev-parse --abbrev-ref HEAD"),
    ("repo_status", "git -C /content/3d-syntree status --short | head -20"),
    ("nvidia_smi", "nvidia-smi"),
    ("free_h", "free -h"),
    ("disk", "df -h /content"),
    ("python_version", "python3 --version"),
    (
        "torch_cuda",
        "python3 - <<'PY'\n"
        "import torch\n"
        "print('torch:', torch.__version__)\n"
        "print('cuda available:', torch.cuda.is_available())\n"
        "print('torch CUDA:', torch.version.cuda)\n"
        "if torch.cuda.is_available():\n"
        "    for i in range(torch.cuda.device_count()):\n"
        "        p = torch.cuda.get_device_properties(i)\n"
        "        print(f'GPU {i}: {p.name} | VRAM {p.total_memory/1024**3:.2f} GiB | cc {p.major}.{p.minor}')\n"
        "PY",
    ),
    (
        "mem_pools",
        "python3 - <<'PY'\n"
        "import psutil, torch, json\n"
        "vm = psutil.virtual_memory()\n"
        "info = {\n"
        "  'host_ram_total_gib': round(vm.total/1024**3, 2),\n"
        "  'host_ram_available_gib': round(vm.available/1024**3, 2),\n"
        "  'process_rss_mib': round(psutil.Process().memory_info().rss/1024**2, 1),\n"
        "}\n"
        "if torch.cuda.is_available():\n"
        "    free, total = torch.cuda.mem_get_info()\n"
        "    info['gpu_free_vram_gib'] = round(free/1024**3, 2)\n"
        "    info['gpu_total_vram_gib'] = round(total/1024**3, 2)\n"
        "print(json.dumps(info, indent=2))\n"
        "PY",
    ),
]


def cmd_verify(client: "paramiko.SSHClient", tag: str) -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    transcript: dict[str, object] = {"tag": tag, "records": {}}
    rc_all = 0
    for name, cmd in VERIFY_CMDS:
        status, text = run(client, cmd, timeout=120.0)
        ok = status == 0
        rc_all |= status
        transcript["records"][name] = {  # type: ignore[index]
            "cmd": cmd,
            "exit": status,
            "ok": ok,
            "output": text.strip(),
        }
        print(f"--- {name} [{'ok' if ok else 'FAIL'}] ---")
        print(text.strip())
    # Policy gate: T4 check (#45 §29.2 — any other device stops GPU work)
    records = transcript["records"]  # type: ignore[index]
    nvidia = str(records["nvidia_smi"]["output"])  # type: ignore[index]
    is_t4 = "Tesla T4" in nvidia
    transcript["is_tesla_t4"] = is_t4  # type: ignore[index]
    print(f"--- policy: Tesla T4 detected: {is_t4} ---")
    if not is_t4:
        print(
            "[colab_ssh_client] HARDWARE POLICY: not a Tesla T4 — GPU "
            "training must NOT proceed (#45 §29.2)."
        )
        rc_all |= 8
    out = EVIDENCE_DIR / f"verify_{tag or 'session'}_{int(time.time())}.json"
    out.write_text(json.dumps(transcript, indent=2))
    print(f"--- transcript saved: {out} ---")
    return rc_all


# ---------------------------------------------------------------------------
# exec / tail / put / get
# ---------------------------------------------------------------------------


def cmd_exec(client: "paramiko.SSHClient", cmd: str, timeout: float) -> int:
    status, text = run(client, cmd, timeout=timeout)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    print(f"[colab_ssh_client] exit status: {status}")
    return status


def cmd_tail(
    client: "paramiko.SSHClient", path: str, seconds: float, interval: float = 5.0
) -> int:
    status, init = run(client, f"wc -l < {path} 2>/dev/null || echo -1")
    if init.strip() == "-1":
        _fail(f"remote file not found: {path}")
    seen = int(init.strip())
    deadline = time.time() + seconds
    while time.time() < deadline:
        status, text = run(
            client,
            f"tail -n +{seen + 1} {path} 2>/dev/null | head -200",
            timeout=60.0,
        )
        lines = text.strip().splitlines()
        if lines:
            seen += len(lines)
            for line in lines:
                print(line)
        time.sleep(interval)
    return 0


def cmd_put(client: "paramiko.SSHClient", local: str, remote: str) -> int:
    sftp = client.open_sftp()
    try:
        sftp.put(local, remote)
        print(f"put {local} -> {remote} ok")
    finally:
        sftp.close()
    return 0


def _recursive_get(sftp: "paramiko.SFTPClient", remote: str, local: str) -> None:
    st = sftp.stat(remote)
    if str(st).startswith("d") or (st.st_mode and (st.st_mode & 0o040000)):
        Path(local).mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote):
            _recursive_get(
                sftp, f"{remote}/{entry.filename}", f"{local}/{entry.filename}"
            )
    else:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, local)


def cmd_get(client: "paramiko.SSHClient", remote: str, local: str) -> int:
    sftp = client.open_sftp()
    try:
        try:
            _recursive_get(sftp, remote, local)
        except FileNotFoundError:
            _fail(f"remote path not found: {remote}")
    finally:
        sftp.close()
    print(f"get {remote} -> {local} ok")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_conn(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--host", default="", help="tunnel host (e.g. 0.tcp.ngrok.io); "
            "empty when --proxy is used"
        )
        p.add_argument("--port", type=int, default=0)
        p.add_argument("--user", default=DEFAULT_USER)
        p.add_argument("--key", default=DEFAULT_KEY)
        p.add_argument(
            "--proxy",
            default=None,
            help="OpenSSH ProxyCommand bridge to ride instead of host:port "
            '(e.g. colab ssh --proxy-mode -s NAME -i KEY)',
        )

    p_v = sub.add_parser("verify", help="§29.2 preflight battery + repo state")
    add_conn(p_v)
    p_v.add_argument("--tag", default="", help="evidence tag (e.g. s-v2-21)")

    p_e = sub.add_parser("exec", help="run a remote command")
    add_conn(p_e)
    p_e.add_argument("command")
    p_e.add_argument("--timeout", type=float, default=600.0)

    p_t = sub.add_parser("tail", help="follow a remote log")
    add_conn(p_t)
    p_t.add_argument("path")
    p_t.add_argument("--seconds", type=float, default=60.0)

    p_p = sub.add_parser("put", help="upload a file")
    add_conn(p_p)
    p_p.add_argument("local")
    p_p.add_argument("remote")

    p_g = sub.add_parser("get", help="download a file/directory (recursive)")
    add_conn(p_g)
    p_g.add_argument("remote")
    p_g.add_argument("local")

    args = ap.parse_args(argv)
    if not args.proxy and not args.host:
        ap.error("--host/--port or --proxy is required")
    client = connect(
        args.host, args.port, args.user, args.key, proxy_command=args.proxy
    )
    try:
        if args.cmd == "verify":
            return cmd_verify(client, args.tag)
        if args.cmd == "exec":
            return cmd_exec(client, args.command, args.timeout)
        if args.cmd == "tail":
            return cmd_tail(client, args.path, args.seconds)
        if args.cmd == "put":
            return cmd_put(client, args.local, args.remote)
        if args.cmd == "get":
            return cmd_get(client, args.remote, args.local)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
