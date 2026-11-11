"""Beginner Cartesian examples: target semantics and virtual CAN only."""

import contextlib
import io
import math
import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from tests.slaves.nero_can_slave import NeroCanSlave
from nero_move_linear import FeedbackUnavailableError, flange_pose, move_linear_offset, move_linear_target
from nero_move_joint import check_before_move, wait_for_motion_feedback
from nero_move_linear import main as linear_main
from nero_move_to import main, make_target, wait_for_flange_pose


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.start = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]

    def test_absolute_mm_and_preserved_orientation(self):
        target, distance = make_target(self.start, [110, 190, 305])
        self.assertEqual(target[:3], [0.11, 0.19, 0.305])
        self.assertEqual(target[3:], self.start[3:])
        self.assertAlmostEqual(distance, 15.0)
        self.assertEqual(self.start[:3], [0.1, 0.2, 0.3])

    def test_omitted_axes_hold_displayed_coordinates(self):
        target, distance = make_target(self.start, [None, None, 305])
        self.assertEqual(target[:3], [0.1, 0.2, 0.305])
        self.assertAlmostEqual(distance, 5.0)

    def test_euclidean_distance_limit_and_explicit_limit(self):
        with self.assertRaises(ValueError):
            make_target(self.start, [120, 220, 300])
        _, distance = make_target(self.start, [120, 220, 300], max_distance_mm=50)
        self.assertAlmostEqual(distance, math.sqrt(800))

    def test_nonfinite_coordinates_and_invalid_limits(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_target(self.start, [value, None, None])
        for limit in (0, -1, float("nan"), float("inf")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                make_target(self.start, [None, None, 305], limit)


class CartesianSlave(NeroCanSlave):
    """Echo pose targets as feedback; no physics or reachability simulation."""

    def __init__(self, channel):
        def pair(a, b):
            return a.to_bytes(4, "big", signed=True) + b.to_bytes(4, "big", signed=True)
        self.pose_data = {
            0x152: pair(100000, -200000),
            0x153: pair(400000, 10000),
            0x154: pair(-20000, 30000),
        }
        self.accept_motion = True
        super().__init__(channel)
        self._joints_enabled = [True] * 7

    def _apply_host_command(self, aid, data):
        super()._apply_host_command(aid, data)
        if aid in self.pose_data and self.accept_motion:
            with self._state_lock:
                self.pose_data[aid] = bytes(data)
                self._proactive_refresh = True

    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        mapping = {0x2A2: 0x152, 0x2A3: 0x153, 0x2A4: 0x154}
        with self._state_lock:
            poses = dict(self.pose_data)
        for frame in frames:
            if frame.arbitration_id in mapping:
                frame.data = bytearray(poses[mapping[frame.arbitration_id]])
        return frames


class UnreachedStartSlave(CartesianSlave):
    def __init__(self, channel):
        self.motion_received = False
        self.report_reached = True
        super().__init__(channel)

    def _apply_host_command(self, aid, data):
        super()._apply_host_command(aid, data)
        if aid in self.pose_data and self.accept_motion:
            with self._state_lock:
                self.motion_received = True

    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        with self._state_lock:
            reached = self.motion_received and self.report_reached
        for frame in frames:
            if frame.arbitration_id == 0x2A1:
                frame.data[4] = 0 if reached else 1
        return frames


class PartialFeedbackSlave(CartesianSlave):
    def __init__(self, channel):
        self.blocked_ids = set()
        super().__init__(channel)

    def _send_and_record(self, msg):
        if msg.arbitration_id not in self.blocked_ids:
            super()._send_and_record(msg)


class VirtualRobotCase(unittest.TestCase):
    slave_type = CartesianSlave

    def setUp(self):
        channel = "absolute_xyz_test_" + uuid.uuid4().hex
        self.device = self.slave_type(channel)
        self.device.start()
        self.robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.V112,
            interface="virtual", channel=channel,
        ))
        self.stop = threading.Event()
        self.feed_paused = threading.Event()
        self.feeder = threading.Thread(target=self.feed, daemon=True)
        self.feeder.start()
        self.robot.connect()
        deadline = time.monotonic() + 2
        while self.robot.get_flange_pose() is None or not all(self.robot.get_joints_enable_status_list()):
            if time.monotonic() > deadline:
                self.fail("Virtual feedback startup timed out")
            time.sleep(0.02)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def feed(self):
        while not self.stop.wait(0.02):
            if not self.feed_paused.is_set():
                self.device.emit_proactive_feedback_once()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.stop.set()
        self.feeder.join(timeout=1)
        self.robot.disconnect()
        self.device.stop()


class VirtualCartesianTests(VirtualRobotCase):
    def test_multi_axis_destination_encodes_absolute_xyz(self):
        start = list(self.robot.get_flange_pose().msg)
        target, _ = make_target(start, [103, -198, 405])
        result = move_linear_target(self.robot, target)
        for actual, expected in zip(result, target):
            self.assertAlmostEqual(actual, expected, places=6)
        frames = {f.arbitration_id: bytes(f.data) for f in self.device.host_frames}
        self.assertEqual(int.from_bytes(frames[0x152][:4], "big", signed=True), 103000)
        self.assertEqual(int.from_bytes(frames[0x152][4:], "big", signed=True), -198000)
        self.assertEqual(int.from_bytes(frames[0x153][:4], "big", signed=True), 405000)
        self.assertEqual(frames[0x153][4:], (10000).to_bytes(4, "big", signed=True))
        self.assertEqual(frames[0x154], (-20000).to_bytes(4, "big", signed=True) + (30000).to_bytes(4, "big", signed=True))
        self.assertFalse(set(frames) & {0x155, 0x156, 0x157, 0x170, 0x471})

    def test_existing_relative_command_still_works(self):
        start = list(self.robot.get_flange_pose().msg)
        result = move_linear_offset(self.robot, "z", 5)
        self.assertAlmostEqual(result[2], start[2] + 0.005)
        for index in (0, 1, 3, 4, 5):
            self.assertAlmostEqual(result[index], start[index])

    def test_missing_pose_rejects_before_transmitting(self):
        with patch.object(self.robot, "get_flange_pose", return_value=None):
            with self.assertRaises(RuntimeError):
                move_linear_target(self.robot, [0.103, -0.198, 0.405, 0, 0, 0])
        self.assertFalse(self.device.host_frames)

    def test_missing_arrival_requests_stop(self):
        self.device.accept_motion = False
        start = list(self.robot.get_flange_pose().msg)
        target, _ = make_target(start, [None, None, 405])
        with self.assertRaises(RuntimeError):
            move_linear_target(self.robot, target, timeout=0.2)
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() > deadline:
                self.fail("No electronic-stop request observed")
            time.sleep(0.01)

    def test_preview_sends_no_motion_or_enable(self):
        with patch("nero_move_to.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_move_to.py", "--x", "103", "--z", "405", "--preview"]):
                self.assertEqual(main(), 0)
        self.assertFalse(self.device.host_frames)


class InteractiveInputTests(VirtualRobotCase):
    def reconnect(self, *args):
        self.connect_calls += 1
        if self.connect_calls == 2:
            self.on_reconnect()
        self.robot.connect()
        return self.robot, "1.12", "v112"

    def setUp(self):
        super().setUp()
        self.connect_calls = 0
        self.on_reconnect = lambda: None

    def test_delayed_input_reopens_and_preserves_fresh_orientation(self):
        answers = iter(("103", "-198", "405"))
        def answer(prompt):
            self.assertFalse(self.robot.is_connected())
            if "X" in prompt:
                time.sleep(0.6)  # Longer than the 0.5-second freshness limit.
            return next(answers)
        def change_orientation():
            with self.device._state_lock:
                self.device.pose_data[0x154] = (-20000).to_bytes(4, "big", signed=True) + (35000).to_bytes(4, "big", signed=True)
        self.on_reconnect = change_orientation
        with patch("nero_move_to.connect_nero", side_effect=self.reconnect):
            with patch("builtins.input", side_effect=answer):
                with patch("sys.argv", ["nero_move_to.py"]):
                    self.assertEqual(main(), 0)
        self.assertEqual(self.connect_calls, 2)
        frames = {f.arbitration_id: bytes(f.data) for f in self.device.host_frames}
        self.assertEqual(int.from_bytes(frames[0x152][:4], "big", signed=True), 103000)
        self.assertEqual(int.from_bytes(frames[0x153][:4], "big", signed=True), 405000)
        self.assertEqual(int.from_bytes(frames[0x154][4:], "big", signed=True), 35000)
        self.assertNotIn(0x471, frames)

    def test_feedback_wait_recovers_without_using_stale_pose(self):
        old_timestamp = self.robot.get_flange_pose().timestamp
        self.feed_paused.set()
        time.sleep(0.6)
        with self.assertRaises(FeedbackUnavailableError):
            flange_pose(self.robot)
        timer = threading.Timer(0.1, self.feed_paused.clear)
        timer.start()
        try:
            _, timestamp = wait_for_flange_pose(self.robot, newer_than=time.time(), timeout=0.5)
            self.assertGreater(timestamp, old_timestamp)
            self.assertLess(time.time() - timestamp, 0.5)
        finally:
            timer.join(timeout=1)
        self.assertFalse(self.device.host_frames)

    def test_no_feedback_after_input_does_not_send_targets(self):
        self.on_reconnect = self.feed_paused.set
        with patch("nero_move_to.connect_nero", side_effect=self.reconnect):
            with patch("builtins.input", side_effect=("103", "-198", "405")):
                with patch("sys.argv", ["nero_move_to.py"]):
                    self.assertEqual(main(), 1)
        self.assertEqual(self.connect_calls, 2)
        self.assertFalse(self.device.host_frames)

    def test_drift_during_input_rechecks_distance_limit(self):
        def change_position():
            with self.device._state_lock:
                self.device.pose_data[0x152] = (150000).to_bytes(4, "big", signed=True) + (-200000).to_bytes(4, "big", signed=True)
        self.on_reconnect = change_position
        with patch("nero_move_to.connect_nero", side_effect=self.reconnect):
            with patch("builtins.input", side_effect=("103", "-198", "405")):
                with patch("sys.argv", ["nero_move_to.py"]):
                    self.assertEqual(main(), 1)
        self.assertFalse(self.device.host_frames)

    def test_cli_coordinates_skip_input_and_use_one_session(self):
        with patch("nero_move_to.connect_nero", side_effect=self.reconnect):
            with patch("builtins.input", side_effect=AssertionError("No input expected")):
                with patch("sys.argv", ["nero_move_to.py", "--x", "103", "--y", "-198", "--z", "405"]):
                    self.assertEqual(main(), 0)
        self.assertEqual(self.connect_calls, 1)
        self.assertTrue(any(frame.arbitration_id == 0x153 for frame in self.device.host_frames))


class PartialFeedbackTests(VirtualRobotCase):
    slave_type = PartialFeedbackSlave

    def test_move_to_waits_for_joint_and_status_after_fresh_pose(self):
        self.device.blocked_ids = {0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A9}
        timer = threading.Timer(1.4, self.device.blocked_ids.clear)
        timer.start()
        try:
            with patch("nero_move_to.connect_nero", return_value=(self.robot, "1.12", "v112")):
                with patch("sys.argv", ["nero_move_to.py", "--x", "103", "--y", "-198", "--z", "405"]):
                    self.assertEqual(main(), 0)
            frames = {frame.arbitration_id: bytes(frame.data) for frame in self.device.host_frames}
            self.assertEqual(int.from_bytes(frames[0x153][:4], "big", signed=True), 405000)
            self.assertNotIn(0x471, frames)
            self.assertNotIn(0x150, frames)
        finally:
            timer.join(timeout=2)

    def test_missing_status_reports_which_stream_and_sends_nothing(self):
        self.device.blocked_ids = {0x2A1}
        time.sleep(0.6)
        flange_pose(self.robot)  # A fresh pose does not imply a fresh status.
        with self.assertRaises(FeedbackUnavailableError) as raised:
            wait_for_motion_feedback(self.robot, timeout=0.1)
        self.assertIn("로봇 상태", str(raised.exception))
        self.assertFalse(self.device.host_frames)

    def test_missing_joint_pair_is_not_hidden_by_fresh_joint7(self):
        self.device.blocked_ids = {0x2A6}
        time.sleep(0.6)
        self.assertLess(time.time() - self.robot.get_joint_angles().timestamp, 0.5)
        with self.assertRaises(FeedbackUnavailableError) as raised:
            wait_for_motion_feedback(self.robot, timeout=0.1)
        self.assertIn("J3/J4", str(raised.exception))
        self.assertFalse(self.device.host_frames)

    def test_missing_driver_waits_for_recovery(self):
        self.device.blocked_ids = {0x263}
        time.sleep(0.6)
        timer = threading.Timer(0.1, self.device.blocked_ids.clear)
        timer.start()
        try:
            angles = check_before_move(self.robot)
            self.assertEqual(len(angles), 7)
        finally:
            timer.join(timeout=1)
        self.assertFalse(self.device.host_frames)

    def test_missing_pose_pair_is_not_hidden_by_fresh_orientation(self):
        self.device.blocked_ids = {0x2A2}
        time.sleep(0.6)
        self.assertLess(time.time() - self.robot.get_flange_pose().timestamp, 0.5)
        with self.assertRaises(FeedbackUnavailableError) as raised:
            flange_pose(self.robot)
        self.assertIn("X/Y", str(raised.exception))
        self.assertFalse(self.device.host_frames)


class UnreachedStartRegressionTests(VirtualCartesianTests):
    slave_type = UnreachedStartSlave

    def test_linear_cli_sends_minus_4mm_from_stable_unreached_start(self):
        self.assertEqual(self.robot.get_arm_status().msg.motion_status, 1)
        with patch("nero_move_linear.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_move_linear.py", "--axis", "z", "--mm", "-4"]):
                self.assertEqual(linear_main(), 0)
        frames = {f.arbitration_id: bytes(f.data) for f in self.device.host_frames}
        self.assertEqual(int.from_bytes(frames[0x153][:4], "big", signed=True), 396000)
        self.assertIn(0x152, frames)
        self.assertIn(0x154, frames)
        self.assertEqual(frames[0x151][1], 2)  # MOVE_L
        self.assertEqual(frames[0x151][2], 5)  # Speed percentage
        self.assertNotIn(0x471, frames)
        self.assertNotIn(0x150, frames)

    def test_changing_joint_feedback_blocks_before_sending(self):
        original = self.robot.get_joint_angles
        started = time.monotonic()
        def moving_angles():
            values = list(original().msg)
            values[0] += math.radians(time.monotonic() - started)
            return SimpleNamespace(msg=values, timestamp=time.time())
        with patch.object(self.robot, "get_joint_angles", side_effect=moving_angles):
            with self.assertRaises(RuntimeError):
                move_linear_offset(self.robot, "z", -4)
        self.assertFalse(self.device.host_frames)

    def test_disabled_motor_blocks_before_sending(self):
        with self.device._state_lock:
            self.device._joints_enabled[2] = False
        time.sleep(0.05)
        with self.assertRaises(RuntimeError):
            move_linear_offset(self.robot, "z", -4)
        self.assertFalse(self.device.host_frames)

    def test_pose_arrival_alone_does_not_override_unreached_feedback(self):
        self.device.report_reached = False
        start = list(self.robot.get_flange_pose().msg)
        target, _ = make_target(start, [None, None, 396])
        with self.assertRaises(RuntimeError) as raised:
            move_linear_target(self.robot, target, timeout=0.2)
        self.assertIn("motion_status=1", str(raised.exception))
        self.assertIn("position_error_mm=0.000", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
