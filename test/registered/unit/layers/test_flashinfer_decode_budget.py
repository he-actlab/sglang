import unittest

from sglang.srt.layers.attention.flashinfer_decode_budget import (
    resolve_decode_planning_width,
)


class TestFlashInferDecodeBudget(unittest.TestCase):
    def test_off_is_stock(self):
        self.assertEqual(
            resolve_decode_planning_width(0, allocated_small_sms=32, physical_sms=108),
            0,
        )

    def test_mode_one_is_exact_partition_width(self):
        self.assertEqual(
            resolve_decode_planning_width(1, allocated_small_sms=32, physical_sms=108),
            32,
        )

    def test_mode_two_is_two_wave_budget(self):
        self.assertEqual(
            resolve_decode_planning_width(2, allocated_small_sms=32, physical_sms=108),
            64,
        )

    def test_mode_two_caps_at_physical_width(self):
        self.assertEqual(
            resolve_decode_planning_width(2, allocated_small_sms=76, physical_sms=108),
            108,
        )

    def test_unsupported_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be 0, 1, or 2"):
            resolve_decode_planning_width(3, allocated_small_sms=32, physical_sms=108)

    def test_invalid_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "positive SM counts"):
            resolve_decode_planning_width(2, allocated_small_sms=0, physical_sms=108)


if __name__ == "__main__":
    unittest.main()
