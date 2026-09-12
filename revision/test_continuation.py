"""检查续训来源、预算、调度和显示；不调用模型、优化器或图像评估。"""

from copy import deepcopy
import unittest

from revision.continuation import comparable_config, continuation_metadata, metric_text, validation_fields
from revision.schedules import continuation_lr_at_step


class ContinuationTests(unittest.TestCase):
    """用人工元数据核对新阶段必须遵守的边界。"""

    def setUp(self):
        """模拟原40k阶段在38k保存的最佳检查点元数据。"""
        self.old = {"model": "full", "max_steps": 40000, "batch_size": 16,
                    "train_manifest_sha256": "train", "val_manifest_sha256": "val",
                    "lr": 1e-4, "warmup_steps": 750, "eval_every": 2000,
                    "scheduler": "legacy_two_stage", "source": {"git": "old"}}
        self.new = {**self.old, "lr": 1e-5, "warmup_steps": 500,
                    "scheduler": "continuation_cosine", "val_lpips": True}
        self.state = {"config": self.old, "step": 38000, "best_val_psnr": 23.76,
                      "optimizer": {"param_groups": [{"lr": 1.1915998492502533e-6}]}}

    def test_phase_is_separate_and_parent_not_mutated(self):
        """新阶段追加40k从38k分支计到78k，原配置和状态保持不变。"""
        original = deepcopy(self.state)
        meta = continuation_metadata(self.state, self.new, "/old/best.pt", "hash")
        self.assertEqual(meta["parent_lineage_step"] + self.new["max_steps"], 78000)
        self.assertEqual(meta["parent_sha256"], "hash")
        self.assertEqual(self.state, original)

    def test_structure_data_and_batch_changes_rejected(self):
        """只允许显式的新预算和调度，不能连带改变数据或结构设置。"""
        for key in ("model", "batch_size", "train_manifest_sha256", "val_manifest_sha256"):
            changed = {**self.new, key: "changed"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation_metadata(self.state, changed, "best.pt", "hash")

    def test_resume_config_compatibility(self):
        """旧版缺省val_lpips等价于False，但改变预算或打开新指标仍需显式新阶段。"""
        self.assertEqual(comparable_config(self.old), comparable_config({**self.old, "val_lpips": False}))
        for change in ({"max_steps": 80000}, {"val_lpips": True}):
            self.assertNotEqual(comparable_config(self.old), comparable_config({**self.old, **change}))

    def test_lr_endpoints_and_direction(self):
        """40k新阶段的前500步平滑升温，随后下降，首尾严格核对。"""
        initial = self.state["optimizer"]["param_groups"][0]["lr"]
        values = [continuation_lr_at_step(s, 40000, 500, 1e-5, initial) for s in range(40000)]
        self.assertEqual(values[0], initial)
        self.assertAlmostEqual(values[499], 1e-5)
        self.assertEqual(values[500], 1e-5)
        self.assertEqual(values[-1], 1e-7)
        self.assertTrue(all(a <= b for a, b in zip(values[:499], values[1:500])))
        self.assertTrue(all(a >= b for a, b in zip(values[500:-1], values[501:])))

    def test_schedule_errors_do_not_silently_change_budget(self):
        """不合法区间或源LR高于峰值时明确拒绝。"""
        for values in ((0, 40000, 1, 1e-5, 1e-6), (40000, 40000, 500, 1e-5, 1e-6),
                       (0, 40000, 500, 1e-5, 1e-4)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                continuation_lr_at_step(*values)

    def test_all_metrics_are_reported_and_missing_is_explicit(self):
        """三个指标按完整验证宏平均输出，不能给旧LPIPS填零。"""
        fields = validation_fields({"groups": {"all": {"psnr": 23.76, "ssim": .82, "lpips": .13}}})
        self.assertIn("SSIM=0.82000", metric_text(fields))
        self.assertIn("LPIPS=0.13000", metric_text(fields))
        del fields["val_macro_lpips"]
        self.assertIn("LPIPS=未测", metric_text(fields))


if __name__ == "__main__":
    unittest.main()
