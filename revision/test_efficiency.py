"""纯Python验证效率协议、计时统计和GPU忙碌守卫；不导入PyTorch或执行模型。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from revision import efficiency


class EfficiencyProtocolTests(unittest.TestCase):
    """通过普通数字、临时文件和假进程验证关键边界。"""

    def test_import_and_plan_do_not_import_torch(self):
        """随机预检计划不创建模型，并且不会传入错误的权重容器键。"""
        self.assertNotIn("torch", sys.modules)
        args = efficiency.parse_args(["plan", "--models", "full", "cidnet", "--allow-untrained"])
        plan = efficiency.make_plan(args)
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual(plan["protocol"]["batch_size"], 1)
        self.assertFalse(plan["protocol"]["tf32"])
        self.assertTrue(all(job["spec"].get("checkpoint_key") is None for job in plan["jobs"]))
        self.assertNotIn("torch", sys.modules)

    def test_missing_weights_need_explicit_preliminary_mode(self):
        """缺少权重不能静默生成看似正式的测量任务。"""
        with self.assertRaisesRegex(ValueError, "缺少权重"):
            efficiency.make_plan(efficiency.parse_args(["plan", "--models", "full"]))

    def test_native_checkpoint_container(self):
        """单模型正式checkpoint自动选model，但纯state_dict必须显式root。"""
        plan = efficiency.make_plan(efficiency.parse_args(["plan", "--checkpoint", "best.pt"]))
        self.assertEqual(plan["models"][0]["checkpoint_key"], "model")
        plan = efficiency.make_plan(efficiency.parse_args(["plan", "--checkpoint", "best.pt", "--checkpoint-key", "root"]))
        self.assertIsNone(plan["models"][0]["checkpoint_key"])

    def test_key_without_checkpoint_rejected_before_run(self):
        """错误的容器键在计划阶段报错，不等到GPU工作开始才发现。"""
        with self.assertRaisesRegex(ValueError, "未指定权重"):
            efficiency.make_plan(efficiency.parse_args(["plan", "--allow-untrained", "--checkpoint-key", "model"]))

    def test_spec_paths_resolve_relative_to_json(self):
        """JSON中的相对权重路径以清单位置为基准，避免当前目录改变时读错文件。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "models.json"
            path.write_text(json.dumps({"models": [{"name": "full", "checkpoint": "weights/best.pt",
                                                     "checkpoint_key": "model"}]}), encoding="utf-8")
            plan = efficiency.make_plan(efficiency.parse_args(["plan", "--spec-file", str(path)]))
            self.assertEqual(Path(plan["models"][0]["checkpoint"]), root / "weights" / "best.pt")

    def test_unknown_structure_field_is_rejected(self):
        """不能误以为清单中的未实现架构参数已经生效。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "models.json"
            path.write_text(json.dumps({"models": [{"name": "full", "refiner_mid_ch": 32}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知字段"):
                efficiency.make_plan(efficiency.parse_args(["plan", "--spec-file", str(path), "--allow-untrained"]))

    def test_latency_statistics_and_invalid_measurements(self):
        """校验分位数、batch1吞吐量，并拒绝NaN或零耗时。"""
        values = efficiency.timing_summary([1., 2., 3., 4.])
        self.assertEqual(values["mean_ms"], 2.5)
        self.assertEqual(values["p50_ms"], 2.5)
        self.assertAlmostEqual(values["p95_ms"], 3.85)
        self.assertEqual(values["images_per_second"], 400.)
        for invalid in ([], [0.], [-1.], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                efficiency.timing_summary(invalid)

    def test_busy_gpu_rejected_without_torch(self):
        """遇到训练进程立即停止，只有本进程可在测量中被排除。"""
        with patch.object(efficiency, "query_compute_processes", return_value=[{"pid": 123, "name": "train.py"}]):
            with self.assertRaisesRegex(RuntimeError, "其他计算任务"):
                efficiency.require_idle_gpu()
            efficiency.require_idle_gpu(own_pid=123)
            with self.assertRaises(RuntimeError):
                efficiency.run_worker({}, Path("must_not_be_created"))
        self.assertNotIn("torch", sys.modules)

    def test_plan_main_does_not_write_output(self):
        """给plan指定output也只打印计划，不写文件或查询GPU。"""
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary) / "not_created"
            with patch.object(efficiency, "query_compute_processes", side_effect=AssertionError("不得查询GPU")):
                code = efficiency.main(["plan", "--allow-untrained", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_partial_flops_are_exported_with_status(self):
        """缺失的FLOPs留空并保留partial标记，不能填零伪装成功。"""
        plan = efficiency.make_plan(efficiency.parse_args(["plan", "--sizes", "256", "--allow-untrained"]))
        report = {"status": "needs_complexity_review", "complexity_per_input": [{"status": "partial",
                  "gflops_counted": None}], "parameters_total": 1000}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            efficiency.export_summary(output, plan, [report])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(summary["rows"][0]["GFLOPs_counted_observed_min"])
            self.assertEqual(summary["rows"][0]["complexity_status"], "partial")
            self.assertFalse(summary["paper_ready"])
            self.assertEqual(summary["status"], "incomplete_or_needs_review")


if __name__ == "__main__":
    unittest.main()
