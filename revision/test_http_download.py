"""用本机 HTTP 服务验证分段范围、断点续传、组装和临时文件清理。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from revision.http_download import ranged_download, cleanup_parts

PAYLOAD = bytes(range(256))*4


class RangeHandler(BaseHTTPRequestHandler):
    """只提供测试字节流和精确的 Range 响应。"""

    def do_GET(self):
        """解析请求的闭区间并返回对应字节。"""
        start, end = map(int, self.headers["Range"].replace("bytes=", "").split("-"))
        body = PAYLOAD[start:end+1]
        self.send_response(206)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, len(PAYLOAD)))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """省略本机测试请求日志。"""
        return


class HTTPDownloadTests(unittest.TestCase):
    """测试只写临时目录，不访问互联网或真实数据集。"""

    def test_resumed_parallel_download_matches_source(self):
        """已有部分分段和临时片段可继续下载，组装内容逐字节一致。"""
        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)/"archive.zip"
                parts = target.parent/"http_parts"/target.name
                parts.mkdir(parents=True)
                (parts/"00000.part").write_bytes(PAYLOAD[:128])
                (parts/"00001.tmp").write_bytes(PAYLOAD[128:133])
                ranged_download("http://127.0.0.1:%d/archive" % server.server_port, target,
                                len(PAYLOAD), workers=3, chunk_size=128)
                self.assertEqual(target.read_bytes(), PAYLOAD)
                cleanup_parts(target)
                self.assertEqual(list(parts.iterdir()), [])
                self.assertEqual(target.read_bytes(), PAYLOAD)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
