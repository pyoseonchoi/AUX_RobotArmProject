"""Regressions for coarse Python 3.12 time vs precise Windows CAN timestamps."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nero_feedback_clock as clock
from nero_read_status import FeedbackUnavailableError, recent, require_feedback


class FeedbackClockTests(unittest.TestCase):
    def test_native_filetime_uses_precise_clock_and_unix_epoch(self):
        def precise(pointer):
            pointer._obj.value = clock._FILETIME_UNIX_EPOCH + 1000002000
        with patch.object(clock, "_precise_filetime", precise):
            with patch.object(clock, "_fallback_time", side_effect=AssertionError("Coarse clock used")):
                self.assertAlmostEqual(clock.feedback_time(), 100.0002, places=7)

    def test_fallback_clock_on_other_platforms(self):
        with patch.object(clock, "_precise_filetime", None):
            with patch.object(clock, "_fallback_time", return_value=123.25):
                self.assertEqual(clock.feedback_time(), 123.25)

    def test_precise_receive_stamp_ahead_of_coarse_clock_is_fresh(self):
        feedback = SimpleNamespace(timestamp=100.00015)
        self.assertLess(100.0 - feedback.timestamp, 0)  # Old coarse-clock failure.
        with patch("nero_read_status.feedback_time", return_value=100.0002):
            self.assertTrue(recent(feedback))
            require_feedback([("J3/J4", feedback)])

    def test_timestamp_is_copied_before_clock_is_read(self):
        now = [100.0]
        class UpdatedFeedback:
            @property
            def timestamp(self):
                now[0] = 100.0001
                return 100.0001
        with patch("nero_read_status.feedback_time", side_effect=lambda: now[0]):
            self.assertTrue(recent(UpdatedFeedback()))
            now[0] = 100.0
            require_feedback([("joint", UpdatedFeedback())])

    def test_stale_data_remains_rejected(self):
        with patch("nero_read_status.feedback_time", return_value=100.0):
            self.assertTrue(recent(SimpleNamespace(timestamp=99.5001)))
            for timestamp in (99.5, 98.0):
                with self.subTest(timestamp=timestamp):
                    self.assertFalse(recent(SimpleNamespace(timestamp=timestamp)))
                    with self.assertRaises(FeedbackUnavailableError):
                        require_feedback([("joint", SimpleNamespace(timestamp=timestamp))])

    def test_real_future_clock_errors_are_identified(self):
        with patch("nero_read_status.feedback_time", return_value=100.0):
            self.assertFalse(recent(SimpleNamespace(timestamp=100.1)))
            with self.assertRaises(FeedbackUnavailableError) as raised:
                require_feedback([("joint", SimpleNamespace(timestamp=100.1))])
        self.assertIn("시계 불일치", str(raised.exception))

    def test_missing_and_invalid_timestamps_are_rejected(self):
        self.assertFalse(recent(None))
        with self.assertRaises(FeedbackUnavailableError):
            require_feedback([("joint", None)])
        for timestamp in (float("nan"), float("inf")):
            self.assertFalse(recent(SimpleNamespace(timestamp=timestamp)))
            with self.assertRaises(FeedbackUnavailableError):
                require_feedback([("joint", SimpleNamespace(timestamp=timestamp))])

    def test_new_session_still_requires_a_new_frame(self):
        feedback = SimpleNamespace(timestamp=99.99)
        with patch("nero_read_status.feedback_time", return_value=100.0):
            with self.assertRaises(FeedbackUnavailableError):
                require_feedback([("joint", feedback)], newer_than=99.995)


if __name__ == "__main__":
    unittest.main()
