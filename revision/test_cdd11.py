"""验证抽样可复现、场景隔离、配对检查和安全解压。"""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
from revision.cdd11 import TASKS, extract_archive, inspect_split, prepare, select_scenes


class CDD11Tests(unittest.TestCase):
    """使用临时合成图像检验数据协议，不访问真实训练数据。"""

    def make_fixture(self, root):
        """生成不同清晰内容及对应退化；不同官方划分允许文件名相同。"""
        for split, count, offset in (("train", 5, 10), ("test", 2, 100)):
            for task in ("clear",) + TASKS:
                directory = root / split / task
                directory.mkdir(parents=True)
                for i in range(count):
                    Image.new("RGB", (16, 16), (offset+i, 20, 30)).save(directory / ("%06d.png" % i))

    def test_stable_group_split(self):
        """相同种子不受文件枚举顺序影响，三个集合互斥且覆盖全部场景。"""
        names = [str(i) for i in range(20)]
        train, val, unused = select_scenes(names, 10, 4, 42)
        self.assertEqual((train, val, unused), select_scenes(list(reversed(names)), 10, 4, 42))
        self.assertEqual(len(set(train + val + unused)), 20)
        self.assertEqual((len(train), len(val), len(unused)), (10, 4, 6))

    def test_manifest_counts_and_no_overwrite(self):
        """检查两种任务设置数量、验证来源及禁止覆盖既有划分。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "official"
            self.make_fixture(root)
            out = Path(tmp) / "manifest"
            summary = prepare(root, out, 3, 1, 42, expected_counts=(5, 2))
            self.assertEqual(summary["files"]["all11_train.json"]["pairs"], 33)
            self.assertEqual(summary["files"]["single4_test.json"]["pairs"], 8)
            train = json.loads((out/"all11_train.json").read_text(encoding="utf-8"))["records"]
            val = json.loads((out/"all11_val.json").read_text(encoding="utf-8"))["records"]
            self.assertFalse({r["target_sha256_rgb"] for r in train} & {r["target_sha256_rgb"] for r in val})
            self.assertTrue(all(r["input"].startswith("train/") for r in val))
            with self.assertRaises(FileExistsError):
                prepare(root, out, 3, 1, expected_counts=(5, 2))

    def test_missing_pair_rejected(self):
        """某一种退化少一张图片时必须报错，禁止按排序截短后配对。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_fixture(root)
            (root/"train"/"rain"/"000000.png").unlink()
            with self.assertRaises(ValueError):
                inspect_split(root/"train")

    def test_clean_content_leakage_rejected(self):
        """即使文件路径不同，训练和测试出现同一清晰内容也必须报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"official"
            self.make_fixture(root)
            Image.open(root/"train"/"clear"/"000000.png").save(root/"test"/"clear"/"000000.png")
            with self.assertRaisesRegex(ValueError, "相同清晰"):
                prepare(root, Path(tmp)/"out", 3, 1, expected_counts=(5, 2))

    def test_zip_traversal_rejected_before_writing(self):
        """先检查整个压缩包，存在越界条目时不写入其中的正常文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/"bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("normal.txt", "normal")
                stream.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_archive(archive, root/"out")
            self.assertFalse((root/"escape.txt").exists())
            self.assertFalse((root/"out"/"normal.txt").exists())

    def test_valid_zip_extracts(self):
        """正常压缩包可解压到指定目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/"good.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("train/clear/a.txt", "ok")
            extract_archive(archive, root/"out")
            self.assertEqual((root/"out/train/clear/a.txt").read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
