"""验证进度读取不会把预分配空间误报为下载完成。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from revision.cdd11 import write_json
from revision.status import snapshot


class StatusTests(unittest.TestCase):
    """使用小型临时文件检验不同下载阶段的状态。"""

    def test_verified_archive_status(self):
        """完整文件与校验标记同时存在时，状态显示为已下载且已校验。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"downloads").mkdir()
            (root/"downloads/test.zip").write_bytes(b"12345678")
            write_json(root/"test.zip.verified.json", {"sha256": "abc"})
            with patch("revision.status.ARCHIVES", {"test.zip": (8, "abc")}):
                state = snapshot(root)
            self.assertEqual(state["written_bytes"], 8)
            self.assertTrue(state["files"][0]["verified"])
            self.assertFalse(state["files"][0]["extracted"])

    def test_sparse_file_not_counted_as_complete(self):
        """逻辑大小等于整包的稀疏文件，也不能计为完整下载。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/"downloads/.cache/huggingface/download"
            folder.mkdir(parents=True)
            partial = folder/"x.abc.incomplete"
            with partial.open("wb") as stream:
                stream.truncate(1024*1024)
            with patch("revision.status.ARCHIVES", {"test.zip": (1024*1024, "abc")}):
                state = snapshot(root)
            self.assertFalse(state["files"][0]["downloaded"])
            self.assertLess(state["written_bytes"], state["total_bytes"])

    def test_log_urls_are_redacted(self):
        """页面日志不暴露下载服务临时签名地址。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"preparation.log").write_text("retry https://example.com/a?token=secret\n", encoding="utf-8")
            state = snapshot(root)
            self.assertNotIn("secret", "".join(state["log"]))
            self.assertIn("下载服务地址", "".join(state["log"]))


if __name__ == "__main__":
    unittest.main()
