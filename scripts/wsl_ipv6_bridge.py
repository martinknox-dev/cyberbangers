"""WSL2 IPv6-to-IPv4 loopback bridge for Streamlit.

Forwards incoming [::1]:8501 requests from Windows browsers to 127.0.0.1:8501.
Allows 'http://localhost:8501' to work seamlessly under WSL2 mirrored networking.
"""

from __future__ import annotations

import select
import socket
import sys
import threading


def _forward_pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            r, _, x = select.select([src], [], [src], 120)
            if not r or x:
                break
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
        try:
            dst.close()
        except Exception:
            pass


def _handle_connection(client: socket.socket, target_port: int) -> None:
    try:
        target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target.connect(("127.0.0.1", target_port))
    except Exception:
        client.close()
        return

    t1 = threading.Thread(target=_forward_pipe, args=(client, target), daemon=True)
    t2 = threading.Thread(target=_forward_pipe, args=(target, client), daemon=True)
    t1.start()
    t2.start()


def run_bridge(listen_port: int = 8501, target_port: int = 8501) -> None:
    server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("::1", listen_port))
    except Exception as e:
        print(f"Bridge bind failed (port may already be bound to IPv6): {e}")
        sys.exit(1)

    server.listen(128)
    print(f"WSL IPv6 bridge active: [::1]:{listen_port} -> 127.0.0.1:{target_port}")

    while True:
        try:
            client, _ = server.accept()
            threading.Thread(
                target=_handle_connection, args=(client, target_port), daemon=True
            ).start()
        except KeyboardInterrupt:
            break
        except Exception:
            pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_bridge(port, port)
