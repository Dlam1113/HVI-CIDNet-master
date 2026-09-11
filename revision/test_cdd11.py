"""验证抽样可复现、场景隔离、配对检查和安全解压。"""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
from revision.cdd11 import (TASKS, extract_archive, group_clean_content, inspect_split,
                            prepare, select_scenes, sha256_file)


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

    def test_duplicate_content_grouped_without_changing_originals(self):
        """清晰像素相同但编码不同仍归为一组，固定代表抽样且保留全部原图和测试图。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"official"
            self.make_fixture(root)
            # 保持退化文件全部存在；只构造两张编码不同、解码后相同的清晰图。
            with Image.open(root/"train"/"clear"/"000000.png") as image:
                image.save(root/"train"/"clear"/"000004.png", compress_level=0)
            self.assertNotEqual(sha256_file(root/"train"/"clear"/"000000.png"),
                                sha256_file(root/"train"/"clear"/"000004.png"))
            original_hashes = {p.relative_to(root).as_posix(): sha256_file(p)
                               for p in root.rglob("*.png")}
            out = Path(tmp)/"manifest"
            summary = prepare(root, out, 2, 1, 20260910, expected_counts=(5, 2))
            audit = summary["audit"]
            self.assertEqual((audit["official_train_scenes"], audit["unique_train_contents"]), (5, 4))
            self.assertEqual(audit["excluded_duplicate_train_files"], 1)
            self.assertEqual(summary["excluded_duplicate_train_scene_ids"], ["000004.png"])
            self.assertEqual(audit["duplicate_train_groups"][0]["representative"], "000000.png")
            self.assertEqual(audit["duplicate_train_groups"][0]["members"], ["000000.png", "000004.png"])
            self.assertEqual(len(summary["unused_unique_train_scene_ids"]), 1)
            self.assertEqual(len(summary["unused_train_scene_ids"]), 2)
            self.assertFalse(summary["protocol"]["raw_images_modified_or_deleted"])
            self.assertEqual(summary["test_scene_ids"], ["000000.png", "000001.png"])
            split_hashes = {}
            for split, count in (("train", 2), ("val", 1), ("test", 2)):
                records = json.loads((out/("all11_"+split+".json")).read_text(encoding="utf-8"))["records"]
                self.assertEqual(len(records), count*11)
                scene_sets = [{r["scene_id"] for r in records if r["task"] == task} for task in TASKS]
                self.assertTrue(all(scenes == scene_sets[0] for scenes in scene_sets))
                split_hashes[split] = {r["target_sha256_rgb"] for r in records}
            self.assertFalse(split_hashes["train"] & split_hashes["val"])
            self.assertFalse((split_hashes["train"] | split_hashes["val"]) & split_hashes["test"])
            self.assertEqual(original_hashes, {p.relative_to(root).as_posix(): sha256_file(p)
                                              for p in root.rglob("*.png")})
            repeated = prepare(root, Path(tmp)/"repeated", 2, 1, 20260910, expected_counts=(5, 2))
            self.assertEqual(summary, repeated)

    def test_duplicate_group_order_and_unique_capacity(self):
        """分组不受文件枚举顺序影响；抽样容量按唯一内容组而非原始文件数计算。"""
        names = ["c.png", "a.png", "b.png", "d.png"]
        hashes = {"c.png": "same", "a.png": "same", "b.png": "other", "d.png": "same"}
        representatives, duplicates = group_clean_content(names, hashes)
        self.assertEqual((representatives, duplicates), group_clean_content(list(reversed(names)), hashes))
        self.assertEqual(representatives, ["a.png", "b.png"])
        self.assertEqual(duplicates[0]["excluded_names"], ["c.png", "d.png"])
        with self.assertRaisesRegex(ValueError, "唯一清晰内容"):
            select_scenes(representatives, 2, 1, 20260910)

    def test_official_raw_count_is_checked_before_sampling(self):
        """存在重复组时仍核验官方原始文件数，不能把去重后数量误作官方样本数。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"official"
            self.make_fixture(root)
            with Image.open(root/"train"/"clear"/"000000.png") as image:
                image.save(root/"train"/"clear"/"000004.png")
            with self.assertRaisesRegex(ValueError, "官方场景数"):
                prepare(root, Path(tmp)/"out", 2, 1, expected_counts=(4, 2))
            self.assertFalse((Path(tmp)/"out").exists())

    def test_duplicate_test_content_not_silently_removed(self):
        """测试集如出现重复内容应停止核实，不能自动删图改变官方测试协议。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"official"
            self.make_fixture(root)
            with Image.open(root/"test"/"clear"/"000000.png") as image:
                image.save(root/"test"/"clear"/"000001.png")
            with self.assertRaisesRegex(ValueError, "不会自动删减"):
                prepare(root, Path(tmp)/"out", 2, 1, expected_counts=(5, 2))
            self.assertFalse((Path(tmp)/"out").exists())

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
