"""在已安装 PyTorch 的服务器验证成对增强、断点采样和指标。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from revision.cdd11 import write_json
from revision.dataset import PairedManifestDataset, StepBatchSampler, assert_no_scene_overlap
from revision.evaluation import aggregate, image_metrics, predict


class PipelineTests(unittest.TestCase):
    """不占用 GPU、不训练实际模型的实验基础设施测试。"""

    def make_dataset(self, root, split="train", digest="a"):
        """生成有纹理的相同输入目标，便于发现裁剪与翻转错位。"""
        pixels = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root/"image.png")
        manifest = root/(split+".json")
        write_json(manifest, {"split": split, "records": [{"scene_id": split+"/a", "task": "rain",
            "input": "image.png", "target": "image.png", "target_sha256_rgb": digest}]})
        return PairedManifestDataset(root, manifest, crop_size=16)

    def test_pair_augmentation_is_aligned(self):
        """输入目标变换一致，且增强只依赖显式种子。"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_dataset(Path(tmp))
            a, b = dataset[(0, 123)], dataset[(0, 123)]
            self.assertTrue(torch.equal(a["input"], a["target"]))
            self.assertTrue(torch.equal(a["input"], b["input"]))
            self.assertEqual(list(a["input"].shape), [3, 16, 16])

    def test_resume_sampling_matches_uninterrupted(self):
        """优化步数恢复后，全部样本编号和增强种子与连续训练相同。"""
        records = [{"task": "rain"}]*3 + [{"task": "snow"}]*7
        complete = list(StepBatchSampler(records, 3, 2, 10, 42))
        resumed = list(StepBatchSampler(records, 3, 2, 10, 42, start_step=4))
        self.assertEqual(complete[8:], resumed)

    def test_test_split_cannot_select_checkpoint(self):
        """即使清晰内容不同，也禁止把 test 当作训练验证集。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self.make_dataset(root, "train", "a")
            test = self.make_dataset(root, "test", "b")
            with self.assertRaises(ValueError):
                assert_no_scene_overlap(train, test)

    def test_content_overlap_rejected(self):
        """不同清单路径不能掩盖相同清晰内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = self.make_dataset(root), self.make_dataset(root, "val")
            with self.assertRaises(ValueError):
                assert_no_scene_overlap(train, val)

    def test_outside_root_is_rejected(self):
        """禁止通过清单相对路径访问数据根目录以外的文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            manifest = root/"train.json"
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            doc["records"][0]["input"] = "../image.png"
            write_json(manifest, doc)
            with self.assertRaises(ValueError):
                PairedManifestDataset(root, manifest)

    def test_padding_preserves_pixels(self):
        """补边后裁回原尺寸，恒等模型保持全部原始像素。"""
        image = torch.rand(1, 3, 19, 27)
        self.assertTrue(torch.equal(predict(torch.nn.Identity(), image), image))

    def test_metric_and_macro_average(self):
        """核对已知误差对应的 PSNR，以及任务等权汇总的权重。"""
        metric = image_metrics(torch.full((3, 16, 16), 0.1), torch.zeros(3, 16, 16))
        self.assertAlmostEqual(metric["psnr"], 20.0, places=5)
        exact = image_metrics(torch.ones(3, 16, 16), torch.ones(3, 16, 16))
        self.assertAlmostEqual(exact["ssim"], 1.0)
        result = aggregate([{"task": "rain", "psnr": 20, "ssim": 0.8}]*2 +
                           [{"task": "snow", "psnr": 30, "ssim": 0.9}])
        self.assertEqual(result["groups"]["all"]["psnr"], 25)

    def test_edge_loss_runs_on_cpu(self):
        """检查设备迁移修正后边缘损失不再隐式申请 CUDA。"""
        from loss.losses import EdgeLoss
        loss = EdgeLoss().to("cpu")
        image = torch.rand(1, 3, 16, 16, requires_grad=True)
        value = loss(image, image.detach())
        value.backward()
        self.assertEqual(value.item(), 0.0)
        self.assertTrue(torch.isfinite(image.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
