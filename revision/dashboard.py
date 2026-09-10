"""仅在 Windows 本机开放的实时进度页面，通过 SSH 只读获取服务器状态。"""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time


class ProgressMonitor:
    """后台定时采样服务器状态，页面请求直接读取缓存而不重复启动 SSH。"""

    def __init__(self, host, project, root, python, interval):
        """保存连接参数与进度缓存；不读取或显示用户密钥。"""
        self.host, self.project, self.root, self.python = host, project, root, python
        self.interval = interval
        self.samples = deque(maxlen=120)
        self.lock = threading.Lock()
        self.data = {"connected": False, "loading": True, "history": []}

    def poll(self):
        """每轮只读查询一次；断连时保留上次进度并明确标记数据陈旧。"""
        remote = "cd %s && %s -B -m revision.status --root %s" % (
            shlex.quote(self.project), shlex.quote(self.python), shlex.quote(self.root))
        while True:
            started = time.monotonic()
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    self.host, remote], capture_output=True, text=True, encoding="utf-8", timeout=22,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
                data = json.loads(result.stdout)
                now = time.time()
                written = data["written_bytes"]
                if self.samples and written < self.samples[-1][1]:
                    self.samples.clear()
                self.samples.append((now, written))
                recent = [sample for sample in self.samples if sample[0] >= now-90]
                rate = ((written-recent[0][1])/(now-recent[0][0])) if len(recent) > 1 else None
                data.update(connected=True, loading=False, observed_at=now, bytes_per_second=rate,
                    eta_seconds=(data["total_bytes"]-written)/rate if rate and rate > 0 else None,
                    history=[{"time": t, "bytes": b} for t, b in self.samples])
                with self.lock:
                    self.data = data
            except Exception as exc:
                with self.lock:
                    self.data = {**self.data, "connected": False, "loading": False,
                                 "connection_error": "服务器状态暂时不可用，将自动重试。",
                                 "error_type": type(exc).__name__}
            time.sleep(max(1, self.interval-(time.monotonic()-started)))

    def read(self):
        """在锁保护下返回不可被页面线程修改的状态副本。"""
        with self.lock:
            return json.loads(json.dumps(self.data))


def handler_for(monitor):
    """生成仅提供页面和只读状态接口的 HTTP 请求处理器。"""
    class Handler(BaseHTTPRequestHandler):
        """限制服务路径与 Host，避免暴露本地项目其他文件。"""

        def do_GET(self):
            """响应首页、图标及状态；其余路径不提供文件访问。"""
            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.send_error(403)
                return
            path = self.path.split("?")[0]
            if path == "/api/status":
                body = json.dumps(monitor.read(), ensure_ascii=False).encode("utf-8")
                mime = "application/json; charset=utf-8"
            elif path == "/":
                body = Path(__file__).with_name("dashboard.html").read_bytes()
                mime = "text/html; charset=utf-8"
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            """省略每次自动刷新日志，避免长期运行产生大量无用输出。"""
            return

    return Handler


def main():
    """启动本地只读进度服务；页面关闭不影响服务器下载。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="Bjj@192.168.5.42")
    parser.add_argument("--project", default="/home/Bjj/HVI-CIDNet-clean")
    parser.add_argument("--root", default="/home/Bjj/HVI-CIDNet-clean/datasets/CDD-11")
    parser.add_argument("--remote-python", default="/home/Bjj/anaconda3/envs/CIDNet/bin/python")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    if args.interval < 3:
        parser.error("状态采样间隔至少 3 秒。")
    monitor = ProgressMonitor(args.host, args.project, args.root, args.remote_python, args.interval)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(monitor))
    threading.Thread(target=monitor.poll, daemon=True).start()
    print("下载进度页面：http://127.0.0.1:%d" % server.server_port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
