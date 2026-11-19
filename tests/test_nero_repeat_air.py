"""Stage-0 recording and motion safeguards; use virtual CAN only."""

import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from nero_move_linear import move_linear_target
from nero_repeat_air import RunRecorder, build_path, main, observe, read_sample, repeat_path
from tests.test_nero_move_to import VirtualRobotCase


class PathTests(unittest.TestCase):
    def test_path_is_anchored_once_and_preserves_orientation(self):
        start = [0.1, -0.2, 0.4, 0.1, -0.2, 0.3]
        a, b = build_path(start, "z", -5)
        self.assertEqual(a, start)
        self.assertAlmostEqual(b[2], 0.395)
        self.assertEqual(b[3:], start[3:])
        b[0] = 0
        self.assertEqual(a, start)

    def test_bad_lengths_or_axes_are_rejected(self):
        for mm in (0, 0.5, 21, -21, float("nan"), float("inf")):
            with self.subTest(mm=mm), self.assertRaises(ValueError):
                build_path([0.0] * 6, "x", mm)
        with self.assertRaises(ValueError):
            build_path([0.0] * 6, "bad", 5)


class AirRecordingTests(VirtualRobotCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "run"
        self.recorder = None

    def tearDown(self):
        if self.recorder is not None and self.recorder.events is not None:
            if not self.recorder.events.closed:
                self.recorder.finish("test_cleanup")
        self.temp.cleanup()
        super().tearDown()

    def start_recorder(self, execute=False):
        self.recorder = RunRecorder(self.robot, self.directory, {"test": True},
                                    enforce_motion_state=execute)
        self.recorder.start()
        return self.recorder

    def test_recording_writes_valid_units_and_packet_timestamps_without_commands(self):
        recorder = self.start_recorder()
        observe(recorder, 0.15)
        summary = recorder.finish("completed")
        with (self.directory / "robot.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(summary["invalid_rows"], 0)
        self.assertAlmostEqual(float(rows[0]["x_mm"]), 100)
        self.assertAlmostEqual(float(rows[0]["roll_deg"]), 10)
        self.assertGreater(float(rows[0]["joint_34_epoch_s"]), 0)
        self.assertGreater(float(rows[0]["end_pose_xy_epoch_s"]), 0)
        self.assertEqual(rows[0]["estimated_speed_mm_s"], "")
        self.assertFalse(self.device.host_frames)

    def test_disabled_motors_can_be_recorded_without_enable(self):
        with self.device._state_lock:
            self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        sample, healthy = read_sample(self.robot)
        self.assertFalse(healthy)
        self.assertEqual(sample["j1_enabled"], 0)
        recorder = self.start_recorder()
        observe(recorder, 0.08)
        summary = recorder.finish("completed")
        self.assertEqual(summary["invalid_rows"], 0)
        self.assertFalse(self.device.host_frames)

    def test_lost_feedback_is_saved_as_invalid_in_record_only_mode(self):
        recorder = self.start_recorder()
        with patch("nero_repeat_air.read_sample", side_effect=RuntimeError("test feedback loss")):
            observe(recorder, 0.12)
        summary = recorder.finish("completed")
        self.assertGreater(summary["invalid_rows"], 0)
        self.assertIn("test feedback loss", (self.directory / "robot.csv").read_text(encoding="utf-8-sig"))
        self.assertFalse(self.device.host_frames)

    def test_execute_recorder_rejects_unhealthy_state_before_any_motion(self):
        with self.device._state_lock:
            self.device._joints_enabled[0] = False
        time.sleep(0.08)
        with self.assertRaises(RuntimeError):
            self.start_recorder(execute=True)
        summary = self.recorder.finish("failed")
        self.assertGreater(summary["invalid_rows"], 0)
        self.assertFalse(self.device.host_frames)

    def test_fixed_endpoints_are_reused_and_results_saved_for_each_leg(self):
        recorder = self.start_recorder(execute=True)
        a, b = build_path(list(self.robot.get_flange_pose().msg), "x", 3)
        repeat_path(self.robot, recorder, a, b, cycles=2, pause=0.05, timeout=1)
        summary = recorder.finish("completed")
        self.assertEqual(len(summary["completed_segments"]), 4)
        self.assertLess(summary["max_endpoint_error_mm"], 0.001)
        values = [int.from_bytes(f.data[:4], "big", signed=True)
                  for f in self.device.host_frames if f.arbitration_id == 0x152]
        self.assertEqual(values, [103000, 100000, 103000, 100000])
        self.assertFalse(any(f.arbitration_id == 0x471 for f in self.device.host_frames))
        events = [json.loads(line) for line in (self.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(e["event"] == "segment_completed" for e in events), 4)

    def test_observer_failure_after_transmission_requests_stop(self):
        target = list(self.robot.get_flange_pose().msg)
        target[0] += 0.003
        calls = []
        def fail_during_motion():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("recorder disk failure")
        with self.assertRaisesRegex(RuntimeError, "recorder disk failure"):
            move_linear_target(self.robot, target, on_feedback=fail_during_motion)
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No emergency stop after recorder failure")
            time.sleep(0.01)

    def test_observer_failure_before_transmission_sends_no_commands(self):
        target = list(self.robot.get_flange_pose().msg)
        target[0] += 0.003
        with self.assertRaisesRegex(RuntimeError, "recording unavailable"):
            move_linear_target(self.robot, target,
                               on_feedback=lambda: self.reject_observer())
        self.assertFalse(self.device.host_frames)

    @staticmethod
    def reject_observer():
        raise RuntimeError("recording unavailable")

    def test_timeout_preserves_partial_recording_and_never_returns_automatically(self):
        self.device.accept_motion = False
        recorder = self.start_recorder(execute=True)
        a, b = build_path(list(self.robot.get_flange_pose().msg), "x", 3)
        with self.assertRaises(RuntimeError):
            repeat_path(self.robot, recorder, a, b, cycles=2, pause=0, timeout=0.15)
        summary = recorder.finish("failed", "arrival timeout")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["completed_segments"], [])
        self.assertGreater(summary["rows"], 0)
        self.assertEqual(sum(f.arbitration_id == 0x152 for f in self.device.host_frames), 1)

    def test_default_cli_is_preview_and_creates_no_files(self):
        with patch("nero_repeat_air.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_repeat_air.py", "--output-dir", self.temp.name]):
                self.assertEqual(main(), 0)
        self.assertFalse(self.device.host_frames)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_record_only_cli_finishes_and_disconnects(self):
        with patch("nero_repeat_air.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_repeat_air.py", "--record-only", "--duration", "0.1",
                                    "--output-dir", self.temp.name]):
                self.assertEqual(main(), 0)
        self.assertFalse(self.robot.is_connected())
        self.assertFalse(self.device.host_frames)
        summaries = list(Path(self.temp.name).glob("*/summary.json"))
        self.assertEqual(len(summaries), 1)
        self.assertEqual(json.loads(summaries[0].read_text(encoding="utf-8"))["status"], "completed")


if __name__ == "__main__":
    unittest.main()
