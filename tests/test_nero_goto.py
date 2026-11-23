"""nero_goto on virtual CAN: preview sends nothing; execution sends joint move, then line."""

import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nero_goto import LineMonitor, main
from nero_kinematics import forward, plan_joint_move, plan_line, pose6
from nero_move_linear import move_linear_target
from tests.conftest import numeric_enum_format
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_teach import TeachingSlave


JOINT_IDS = (0x155, 0x156, 0x157, 0x170)
NEAR = ("--x", "-100", "--z", "650")   # From upright: about 25 degrees of joint move, then a line.
FAR = ("--x", "-350", "--z", "300")
POSE_IDS = (0x152, 0x153, 0x154)


def pair(a, b):
    return int(round(a)).to_bytes(4, "big", signed=True) + int(round(b)).to_bytes(4, "big", signed=True)


class GotoSlave(TeachingSlave):
    """Teleports to joint targets and reports the flange pose of the SDK model."""

    def __init__(self, channel):
        self.pose_shift_m = 0.0
        super().__init__(channel)
        self.follow_joints()

    def follow_joints(self):
        with self._state_lock:
            joints = list(self.current_joints)
        x, y, z, roll, pitch, yaw = pose6(*forward(joints))
        frames = {0x152: pair((x + self.pose_shift_m) * 1e6, y * 1e6),
                  0x153: pair(z * 1e6, math.degrees(roll) * 1e3),
                  0x154: pair(math.degrees(pitch) * 1e3, math.degrees(yaw) * 1e3)}
        with self._state_lock:
            self.pose_data.update(frames)

    def _apply_host_command(self, aid, data):
        super()._apply_host_command(aid, data)
        if aid in JOINT_IDS and self.accept_joints:
            self.follow_joints()


def sent_joints(frames):
    """Decode the last seven-joint target (rad) from host frames."""
    last = {f.arbitration_id: bytes(f.data) for f in frames if f.arbitration_id in JOINT_IDS}
    values = []
    for aid in JOINT_IDS:
        values += [int.from_bytes(last[aid][i:i + 4], "big", signed=True) for i in (0, 4)]
    return [math.radians(v / 1000) for v in values[:7]]


class GotoCase(VirtualRobotCase):
    slave_type = GotoSlave

    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        for target, value in (("nero_goto.START_DELAY_S", 0), ("nero_diagnostics.DEFAULT_ROOT", self.temp.name),
                              ("nero_goto.connect_nero", lambda *a: self.connect())):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.runs = Path(self.temp.name) / "runs"

    def tearDown(self):
        super().tearDown()
        self.temp.cleanup()

    def connect(self):
        self.robot.connect()
        return self.robot, "1.12", "v112"

    def run_goto(self, *options):
        with patch("sys.argv", ["nero_goto.py", "--output-dir", str(self.runs)] + list(options)):
            return main()

    def target(self, x_mm, z_mm):
        """Same target the CLI builds: omitted Y keeps the measured flange Y."""
        return [x_mm / 1000, self.robot.get_flange_pose().msg[1], z_mm / 1000]


class GotoVirtualTests(GotoCase):
    def test_preview_from_upright_sends_nothing(self):
        self.assertEqual(self.run_goto(*FAR), 0)
        self.assertFalse(self.device.host_frames)
        self.assertFalse(self.runs.exists())

    def test_preview_works_while_disabled_but_execute_refuses(self):
        with self.device._state_lock:
            self.device._joints_enabled[2] = False
        time.sleep(0.1)
        self.assertEqual(self.run_goto(*FAR), 0)
        self.assertEqual(self.run_goto(*NEAR, "--execute"), 1)
        self.assertFalse(self.device.host_frames)

    def test_execute_from_upright_moves_joints_together_then_sends_line(self):
        plan = plan_line([0.0] * 7, self.target(-100, 650), self.robot.get_config()["joint_limits"])
        self.assertTrue(plan.needs_joint_move)
        self.assertEqual(self.run_goto(*NEAR, "--execute"), 0)
        frames = self.device.host_frames
        for sent, planned in zip(sent_joints(frames), plan.entry_joints):
            self.assertAlmostEqual(sent, planned, delta=math.radians(0.002))
        poses = {f.arbitration_id: bytes(f.data) for f in frames if f.arbitration_id in POSE_IDS}
        self.assertEqual(int.from_bytes(poses[0x152][:4], "big", signed=True), -100000)
        self.assertEqual(int.from_bytes(poses[0x153][:4], "big", signed=True), 650000)
        # Joint mode first, then linear mode; default speed; no enable, stop or reset.
        modes = [f.data[1] for f in frames if f.arbitration_id == 0x151 and f.data[1] != 255]
        self.assertEqual(sorted(set(modes), key=modes.index), [1, 2])
        self.assertTrue(all(f.data[2] == 7 for f in frames if f.arbitration_id == 0x151))
        # By default the joint target is led forward in steps rather than sent once.
        self.assertGreater(sum(f.arbitration_id == 0x155 for f in frames), 3)
        order = [f.arbitration_id for f in frames if f.arbitration_id in JOINT_IDS + POSE_IDS]
        self.assertLess(max(i for i, aid in enumerate(order) if aid in JOINT_IDS),
                        min(i for i, aid in enumerate(order) if aid in POSE_IDS))
        self.assertFalse(any(f.arbitration_id in (0x150, 0x471) for f in frames))
        run, = self.runs.iterdir()
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertLess(summary["completed_segments"][0]["max_line_deviation_mm"], 0.1)

    def test_joint_path_sends_one_joint_target_at_requested_speed(self):
        plan = plan_joint_move([0.0] * 7, self.target(-350, 300), self.robot.get_config()["joint_limits"])
        self.assertGreater(plan.joint_move_deg, 60)
        options = FAR + ("--path", "joint", "--joint-move", "single", "--speed", "7", "--execute")
        self.assertEqual(self.run_goto(*options, "--max-joint-move-deg", "60"), 1)  # An explicit cap still holds.
        self.assertFalse(self.device.host_frames)
        self.assertEqual(self.run_goto(*options), 0)  # The default no longer caps the joint move.
        frames = self.device.host_frames
        for sent, planned in zip(sent_joints(frames), plan.entry_joints):
            self.assertAlmostEqual(sent, planned, delta=math.radians(0.002))
        self.assertEqual(sum(f.arbitration_id == 0x155 for f in frames), 1)
        self.assertTrue(all(f.data[2] == 7 for f in frames if f.arbitration_id == 0x151))
        self.assertFalse(any(f.arbitration_id in POSE_IDS + (0x150, 0x471) for f in frames))

    def test_unreachable_target_and_bad_options_send_nothing(self):
        self.assertEqual(self.run_goto("--x", "900", "--z", "300", "--execute"), 1)
        with patch("nero_goto.connect_nero", side_effect=AssertionError("Must not connect")):
            for options in (["--execute"], ["--x", "-350", "--speed", "21"], ["--x", "-350", "--speed", "0"],
                            ["--x", "-350", "--max-deviation-mm", "3"], ["--x", "-350", "--max-deviation-mm", "101"],
                            ["--x", "-350", "--max-joint-move-deg", "361"], ["--x", "-350", "--max-joint-move-deg", "0"],
                            ["--x", "-350", "--rpy", "0", "0", "0", "--orient", "keep"],
                            ["--x", "-350", "--rpy", "0", "95", "0"], ["--x", "nan"],
                            ["--x", "-350", "--timeout", "0"], ["--x", "-350", "--min-sigma", "0"]):
                with self.subTest(options=options):
                    self.assertEqual(self.run_goto(*options), 1)
        self.assertFalse(self.device.host_frames)

    def test_model_mismatch_blocks_execution_only(self):
        self.device.pose_shift_m = 0.010
        self.device.follow_joints()
        time.sleep(0.1)
        self.assertEqual(self.run_goto(*FAR), 0)
        self.assertEqual(self.run_goto(*NEAR, "--execute"), 1)
        self.assertFalse(self.device.host_frames)

    def test_feedback_leaving_the_line_requests_stop(self):
        original = self.device._apply_host_command
        def bend(aid, data):
            original(aid, data)
            if aid == 0x152:  # Report a flange 60 mm beside the line: past the default 50 mm stop limit.
                with self.device._state_lock:
                    y = int.from_bytes(self.device.pose_data[0x152][4:], "big", signed=True)
                    self.device.pose_data[0x152] = self.device.pose_data[0x152][:4] + pair(y + 60000, 0)[:4]
        with patch.object(self.device, "_apply_host_command", side_effect=bend):
            self.assertEqual(self.run_goto(*NEAR, "--execute"), 1)
        self.assertTrue(any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames))
        run, = self.runs.iterdir()
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertIn("직선에서", summary["reason"])


    def test_line_after_joint_move_skips_only_the_second_stillness_window(self):
        with patch("nero_move_linear.check_before_move", side_effect=AssertionError("second 0.5 s wait")):
            self.assertEqual(self.run_goto(*NEAR, "--execute"), 0)
        self.assertTrue(any(f.arbitration_id in POSE_IDS for f in self.device.host_frames))

    def test_settled_line_still_refuses_a_disabled_motor(self):
        with self.device._state_lock:
            self.device._joints_enabled[2] = False
        time.sleep(0.1)
        target = list(self.robot.get_flange_pose().msg)
        target[2] -= 0.004
        with self.assertRaisesRegex(RuntimeError, "J3"):
            move_linear_target(self.robot, target, settled=True)
        self.assertFalse(self.device.host_frames)

    def test_controller_rejection_is_named_in_the_failure(self):
        # Replays 2026-11-16: move_l was accepted, never progressed, then NO_SOLUTION.
        original = self.device._apply_host_command
        def reject(aid, data):
            if aid in POSE_IDS:
                self.device.reported_arm_status = 2
                return
            original(aid, data)
        with patch.object(self.device, "_apply_host_command", side_effect=reject), \
             patch.object(type(self.robot.get_arm_status().msg.arm_status), "__format__", numeric_enum_format):
            self.assertEqual(self.run_goto(*NEAR, "--execute"), 1)
        log, = Path(self.temp.name).glob("*.jsonl")
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        failure, = [e for e in events if e["event"] == "failed"]
        self.assertIn("NO_SOLUTION", failure["error"])
        self.assertIn("거부", failure["error"])
        # The controller already gave up and the arm is still: no emergency stop, no reset needed.
        self.assertTrue(any(e["event"] == "emergency_stop_skipped" for e in events))
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))


class LineMonitorTests(unittest.TestCase):
    def test_tracks_worst_deviation_calls_inner_hook_and_stops_past_limit(self):
        pose = SimpleNamespace(msg=[0.1, 0.0, 0.0, 0, 0, 0])
        robot = SimpleNamespace(get_flange_pose=lambda: pose, get_arm_status=lambda: None)
        calls = []
        monitor = LineMonitor(robot, [0, 0, 0], [1, 0, 0], 20.0, also=lambda: calls.append(1))
        monitor()
        pose.msg[1] = 0.012
        monitor()
        pose.msg[1] = 0.004
        monitor()
        self.assertAlmostEqual(monitor.worst_mm, 12.0)
        self.assertEqual(len(calls), 3)
        pose.msg[1] = 0.021
        with self.assertRaisesRegex(RuntimeError, "직선에서"):
            monitor()

    def test_moderate_drift_only_warns_once_and_keeps_going(self):
        pose = SimpleNamespace(msg=[0.1, 0.0, 0.0, 0, 0, 0])
        robot = SimpleNamespace(get_flange_pose=lambda: pose, get_arm_status=lambda: None)
        warnings = []
        monitor = LineMonitor(robot, [0, 0, 0], [1, 0, 0], 50.0, on_warning=warnings.append)
        for drift in (0.010, 0.025, 0.040, 0.030, 0.049):
            pose.msg[1] = drift
            monitor()  # Never raises below the stop limit.
        self.assertEqual(len(warnings), 1)
        self.assertAlmostEqual(warnings[0], 25.0)
        self.assertAlmostEqual(monitor.worst_mm, 49.0)
        pose.msg[1] = 0.051
        with self.assertRaisesRegex(RuntimeError, "정지 한도 50"):
            monitor()

    def test_missing_feedback_is_left_to_the_motion_loop(self):
        monitor = LineMonitor(SimpleNamespace(get_flange_pose=lambda: None, get_arm_status=lambda: None),
                              [0, 0, 0], [1, 0, 0], 20.0)
        monitor()
        self.assertEqual(monitor.worst_mm, 0.0)


if __name__ == "__main__":
    unittest.main()
