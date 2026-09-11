"""仅验证纯标量学习率公式，不导入 PyTorch，不执行模型或训练。"""

import math
import unittest

from revision.schedules import lr_at_step


class ScheduleTests(unittest.TestCase):
    """核对预热、两段端点、重启跳变及旧单余弦兼容性。"""

    def test_legacy_warmup_includes_zero_and_base(self):
        """多步预热的首尾准确覆盖 0 和基础学习率。"""
        values = [lr_at_step(step, 40, 3, 1e-4) for step in range(3)]
        self.assertEqual(values, [0.0, 5e-5, 1e-4])

    def test_legacy_phase_boundaries_and_restart(self):
        """第一段末尾到达 2e-4，下一更新重启到基础值。"""
        self.assertEqual(lr_at_step(3, 40, 3, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(9, 40, 3, 1e-4), 2e-4)
        self.assertEqual(lr_at_step(10, 40, 3, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(39, 40, 3, 1e-4), 1e-7)

    def test_legacy_segments_have_expected_direction(self):
        """基础值为 1e-4 时，第一余弦段上升、第二段下降。"""
        values = [lr_at_step(step, 100, 5, 1e-4) for step in range(100)]
        self.assertTrue(all(a < b for a, b in zip(values[5:24], values[6:25])))
        self.assertTrue(all(a > b for a, b in zip(values[25:99], values[26:100])))

    def test_legacy_without_warmup(self):
        """零预热时第一更新直接使用基础学习率。"""
        self.assertEqual(lr_at_step(0, 40, 0, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(9, 40, 0, 1e-4), 2e-4)

    def test_legacy_one_warmup_update(self):
        """单步预热明确定义为基础值，不进行除零或假装覆盖两个端点。"""
        self.assertEqual(lr_at_step(0, 40, 1, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(1, 40, 1, 1e-4), 1e-4)

    def test_legacy_minimal_first_segment(self):
        """第一段只有两更新仍准确得到起点、峰值和重启点。"""
        self.assertEqual(lr_at_step(0, 8, 0, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(1, 8, 0, 1e-4), 2e-4)
        self.assertEqual(lr_at_step(2, 8, 0, 1e-4), 1e-4)
        self.assertEqual(lr_at_step(7, 8, 0, 1e-4), 1e-7)

    def test_legacy_rejects_insufficient_segment_length(self):
        """不能让预热挤占余弦区间后静默更改学习率预算。"""
        for total, warmup in ((7, 0), (40, 9), (40, 10), (12, 2)):
            with self.subTest(total=total, warmup=warmup), self.assertRaises(ValueError):
                lr_at_step(0, total, warmup, 1e-4)

    def test_legacy_peak_is_old_absolute_eta_min(self):
        """第一段终点沿用旧 eta_min 的绝对 2e-4，不擅自改为两倍 base。"""
        self.assertEqual(lr_at_step(9, 40, 0, 5e-5), 2e-4)
        self.assertEqual(lr_at_step(10, 40, 0, 5e-5), 5e-5)

    def test_single_cosine_matches_previous_formula(self):
        """逐点核对既有单余弦公式，包含零预热、一步预热和最短预算。"""
        for total, warmup in ((1, 0), (2, 0), (10, 0), (10, 1), (10, 3), (10, 9)):
            for step in range(total):
                if step < warmup:
                    expected = 1e-4 * (step + 1) / max(1, warmup)
                else:
                    fraction = (step - warmup) / max(1, total - warmup - 1)
                    expected = 1e-7 + 0.5 * (1e-4 - 1e-7) * (1 + math.cos(math.pi * fraction))
                self.assertEqual(lr_at_step(step, total, warmup, 1e-4, "single_cosine"), expected)

    def test_invalid_step_or_warmup_is_rejected(self):
        """不允许负数索引、预算外索引或非法预热长度。"""
        for step, total, warmup in ((-1, 40, 3), (40, 40, 3), (0, 0, 0), (0, 40, -1), (0, 40, 40)):
            with self.subTest(step=step, total=total, warmup=warmup), self.assertRaises(ValueError):
                lr_at_step(step, total, warmup, 1e-4)

    def test_noninteger_counts_are_rejected(self):
        """更新编号和预算必须是整数，布尔值也不作为有效编号。"""
        for args in ((0.0, 40, 3), (0, 40.0, 3), (0, 40, 3.0), (False, 40, 3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                lr_at_step(*args, 1e-4)

    def test_invalid_base_is_rejected(self):
        """学习率不得为零、负值或非有限值。"""
        for base in (0, -1e-4, float("nan"), float("inf")):
            with self.subTest(base=base), self.assertRaises(ValueError):
                lr_at_step(0, 40, 3, base)

    def test_unknown_schedule_is_rejected(self):
        """拼写错误的调度器名称不能静默退化成默认配置。"""
        with self.assertRaises(ValueError):
            lr_at_step(0, 40, 3, 1e-4, "unknown")


if __name__ == "__main__":
    unittest.main()
