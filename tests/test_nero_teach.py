"""Manual-teaching file validation and mode/replay checks on virtual CAN."""

import csv
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pyAgxArm import ArmModel, create_agx_arm_config
from nero_move_joint import move_joint_target
from nero_read_status import FeedbackUnavailableError, feedback_time
from nero_teach import (FORMAT, execute_path, leader_sample, load_teaching, main,
                        make_waypoints, prepare_joint_mode, record_hand, restore_control, wait_for_leader)
from tests.slaves import _can_payloads as pl
from tests.test_nero_move_to import CartesianSlave, VirtualRobotCase


def sample(t, degrees):
    return dict(t_s=t, q_rad=[math.radians(degrees)] + [0.0] * 6)


def save_fixture(directory, samples=None, **changes):
    path = Path(directory) / "trajectory.json"
    data = dict(format=FORMAT, robot_model="nero", joint_units="rad", status="completed",
                control_restored=True, samples=samples or [sample(0, 0), sample(0.1, 1)])
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TeachingFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.limits = create_agx_arm_config(robot=ArmModel.NERO)["joint_limits"]

    def tearDown(self):
        self.temp.cleanup()

    def test_file_units_model_and_completion_are_mandatory(self):
        for changes in (dict(status="recording"), dict(status="failed"),
                        dict(control_restored=False), dict(robot_model="piper"), dict(joint_units="deg")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_teaching(save_fixture(self.temp.name, **changes), self.limits)

    def test_nan_limits_bad_dimensions_and_time_gaps_are_rejected(self):
        invalid = [
            [sample(0, 0), sample(0.5, 1)], [sample(0, 0), sample(0, 1)],
            [sample(0.2, 0), sample(0.1, 1)], [sample(0, 0), sample(0.1, 200)],
            [sample(0, 0), dict(t_s=0.1, q_rad=[0.0] * 6)],
            [sample(0, 0), dict(t_s=0.1, q_rad=[float("nan")] * 7)],
        ]
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                load_teaching(save_fixture(self.temp.name, samples), self.limits)

    def test_simplification_keeps_endpoints_and_bends_and_bounds_every_step(self):
        samples = [sample(0, 0), sample(0.1, 3), sample(0.2, 6), sample(0.3, 3), sample(0.4, 0)]
        points, span = make_waypoints(samples, step_degrees=2)
        self.assertAlmostEqual(span, 6)
        self.assertEqual(points[0], samples[0]["q_rad"])
        self.assertEqual(points[-1], samples[-1]["q_rad"])
        self.assertAlmostEqual(max(q[0] for q in points), math.radians(6))
        self.assertTrue(all(max(abs(x - y) for x, y in zip(a, b)) <= math.radians(2) + 1e-9
                            for a, b in zip(points, points[1:])))

    def test_straight_path_simplifies_without_changing_source(self):
        samples = [sample(i / 10, i) for i in range(5)]
        points, _ = make_waypoints(samples, step_degrees=4)
        self.assertEqual(len(points), 2)
        self.assertEqual(len(samples), 5)

    def test_span_and_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            make_waypoints([sample(0, 0), sample(0.1, 40)])
        for kwargs in (dict(tolerance_degrees=float("nan")), dict(step_degrees=0), dict(max_span_degrees=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_waypoints([sample(0, 0), sample(0.1, 1)], **kwargs)

    def test_inspect_and_previews_never_connect(self):
        path = save_fixture(self.temp.name)
        with patch("nero_teach.connect_nero", side_effect=AssertionError("Must remain offline")):
            for args in (["inspect", str(path)], ["home", str(path)], ["replay", str(path)],
                         ["record"], ["control"], ["prepare"]):
                with self.subTest(args=args), patch("sys.argv", ["nero_teach.py"] + args):
                    self.assertEqual(main(), 0)


class TeachingSlave(CartesianSlave):
    """Only leader packets during teaching, only ordinary feedback otherwise."""
    def __init__(self, channel):
        self.leader = False
        self.blocked_ids = set()
        self.current_joints = [0.0] * 7
        self.accept_joints = True
        self.entered_at = None
        self.hand_motion = False
        self.jump_on_follower = False
        self.normal_push = True
        self.ignore_leader_request = False
        self.reject_push_enable = False
        self.emit_leader_packets = True
        self.leader_suppresses_normal = False
        self.disable_on_follower = False
        self.reclaim_delay = 0.0
        self.ignore_can_takeover = False
        self.ignore_motion_change = False
        self.pending_can_at = None
        self.reported_arm_status = 0
        super().__init__(channel)
        self._leader_rad = [0.0] * 7

    def _apply_host_command(self, aid, data):
        previous_ctrl, previous_motion = self._ctrl_mode, self._mode_feedback
        super()._apply_host_command(aid, data)
        if aid == 0x151:
            with self._state_lock:
                # Real feedback: role may change yet control mode stays at 6.
                # Reporting-only FF/FF messages must not pretend to reclaim CAN.
                if data[1] == 255 or data[3] == 255 or self.leader or self.ignore_can_takeover:
                    self._ctrl_mode, self._mode_feedback = previous_ctrl, previous_motion
                elif data[0] == 1 and previous_ctrl == 6 and self.reclaim_delay:
                    self._ctrl_mode = previous_ctrl
                    self.pending_can_at = time.monotonic() + self.reclaim_delay
                if self.ignore_motion_change:
                    self._mode_feedback = previous_motion
        if aid == 0x151 and data[6] in (1, 2):
            self.normal_push = data[6] == 1 and not self.reject_push_enable
        if aid == 0x470:
            if data[0] == 0xFA and self.ignore_leader_request:
                return
            with self._state_lock:
                self.leader = data[0] == 0xFA
                if self.leader:
                    self.entered_at = time.monotonic()
                    self._ctrl_mode = 6
                else:
                    self.current_joints = list(self._leader_rad)
                    if self.disable_on_follower:
                        self._joints_enabled = [False] * 7
                    if self.jump_on_follower:
                        self.current_joints[0] += math.radians(2)
        if aid in (0x155, 0x156, 0x157, 0x170) and self.accept_joints:
            index = {0x155: 0, 0x156: 2, 0x157: 4, 0x170: 6}[aid]
            with self._state_lock:
                for offset in range(1 if index == 6 else 2):
                    value = int.from_bytes(data[4 * offset:4 * offset + 4], "big", signed=True)
                    self.current_joints[index + offset] = math.radians(value / 1000)
                self._leader_rad = list(self.current_joints)

    def _master_feedback_messages(self):
        if self.pending_can_at is not None and time.monotonic() >= self.pending_can_at:
            self._ctrl_mode = 1
            self.pending_can_at = None
        if self.leader and self.hand_motion and self.entered_at is not None:
            with self._state_lock:
                self._leader_rad[0] = math.radians(min(0.6, max(0, time.monotonic() - self.entered_at - 0.06) * 6))
        frames = super()._master_feedback_messages()
        if self.leader and (not self.normal_push or self.leader_suppresses_normal):
            if not self.emit_leader_packets:
                return []
            return [f for f in frames if f.arbitration_id in (0x155, 0x156, 0x157, 0x170)]
        if not self.leader or not self.emit_leader_packets:
            frames = [f for f in frames if f.arbitration_id not in (0x155, 0x156, 0x157, 0x170)]
        with self._state_lock:
            joints = list(self._leader_rad if self.leader else self.current_joints)
        for f in frames:
            if f.arbitration_id == 0x2A1:
                f.data[1] = self.reported_arm_status
            if f.arbitration_id in (0x2A5, 0x2A6, 0x2A7):
                index = (f.arbitration_id - 0x2A5) * 2
                # The stock feedback helper returns a fixed captured frame and
                # intentionally ignores its arguments. Encode the actual target.
                f.data = bytearray(pl.pack_joint_ctrl_pair_rad(*joints[index:index + 2]))
            if f.arbitration_id == 0x2A9:
                f.data = bytearray(pl.pack_joint_ctrl_pair_rad(joints[6], 0.0))
        return frames

    def _standard_feedback_burst(self):
        if self.leader and (not self.normal_push or self.leader_suppresses_normal):
            return self._master_feedback_messages()
        if not self.normal_push:
            return []
        return super()._standard_feedback_burst()

    def _send_and_record(self, msg):
        if msg.arbitration_id not in self.blocked_ids:
            super()._send_and_record(msg)


class TeachingVirtualTests(VirtualRobotCase):
    slave_type = TeachingSlave

    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "teaching"
        self.log_root = Path(self.temp.name) / "diagnostics"
        log_patch = patch("nero_teach.DIAGNOSTIC_ROOT", self.log_root)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def tearDown(self):
        self.temp.cleanup()
        super().tearDown()

    def test_record_uses_leader_feedback_and_restores_without_targets_or_disable(self):
        self.device.hand_motion = True
        samples = record_hand(self.robot, self.directory, {"test": True}, duration=0.25)
        self.assertGreater(len(samples), 4)
        self.assertGreater(samples[-1]["q_rad"][0], samples[0]["q_rad"][0])
        _, data = load_teaching(self.directory, self.robot.get_config()["joint_limits"])
        self.assertTrue(data["control_restored"])
        self.assertFalse(self.device.leader)
        frames = self.device.host_frames
        mode_configs = [f.data[0] for f in frames if f.arbitration_id == 0x470]
        self.assertEqual(mode_configs, [0xFA, 0xFC])
        self.assertTrue(self.device.normal_push)
        self.assertTrue(any(f.arbitration_id == 0x151 and f.data[6] == 1 for f in frames))
        self.assertFalse(any(f.arbitration_id == 0x151 and f.data[6] == 2 for f in frames))
        mode_packets = [f for f in frames if f.arbitration_id == 0x151 and f.data[6] == 1]
        self.assertTrue(all(f.data[2] == 5 for f in mode_packets))
        self.assertTrue(any(f.data[0] == 1 and f.data[1] == 1 and f.data[3] == 0 for f in mode_packets))
        self.assertEqual(int(self.robot._msg_mode.enable_can_push), 0)
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471) for f in frames))
        with (self.directory / "joints.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(samples))

    def test_partial_leader_stream_is_not_hidden_by_fresh_joint7(self):
        self.robot.set_leader_mode()
        wait_for_leader(self.robot, newer_than=feedback_time())
        self.device.blocked_ids = {0x156}
        time.sleep(0.6)
        with self.assertRaises(FeedbackUnavailableError) as raised:
            leader_sample(self.robot)
        self.assertIn("leader_joint_34", str(raised.exception))

    def test_record_exits_real_leader_with_no_normal_feedback_until_follower(self):
        self.device.leader_suppresses_normal = True
        self.device.hand_motion = True
        samples = record_hand(self.robot, self.directory, {}, duration=0.65)
        self.assertGreater(len(samples), 10)
        _, data = load_teaching(self.directory, self.robot.get_config()["joint_limits"])
        self.assertTrue(data["control_restored"])
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertFalse(any(f.arbitration_id in (0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_standalone_control_observes_leader_before_requiring_normal_packets(self):
        self.device.leader_suppresses_normal = True
        self.robot.set_leader_mode()
        wait_for_leader(self.robot, newer_than=feedback_time())
        time.sleep(0.6)
        restore_control(self.robot)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertFalse(any(f.arbitration_id in (0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_missing_normal_after_follower_blocks_valid_can_takeover(self):
        self.device.leader_suppresses_normal = True
        self.robot.set_leader_mode()
        wait_for_leader(self.robot, newer_than=feedback_time())
        time.sleep(0.6)
        self.device.blocked_ids = {0x2A1}
        with self.assertRaisesRegex(FeedbackUnavailableError, "일반 피드백 재개"):
            restore_control(self.robot, timeout=0.15)
        self.assertFalse(any(f.arbitration_id == 0x151 and f.data[1] != 255
                             for f in self.device.host_frames))
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_motor_state_change_after_leader_exit_keeps_record_failed(self):
        self.device.leader_suppresses_normal = True
        self.device.disable_on_follower = True
        with self.assertRaisesRegex(RuntimeError, "모터 활성화 상태가 달라져"):
            record_hand(self.robot, self.directory, {}, duration=0.65)
        data = json.loads((self.directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertFalse(data["control_restored"])
        self.assertFalse(any(f.arbitration_id == 0x151 and f.data[1] != 255
                             for f in self.device.host_frames))

    def test_mode_handover_motion_requests_stop_and_marks_record_failed(self):
        self.device.jump_on_follower = True
        with self.assertRaises(RuntimeError):
            record_hand(self.robot, self.directory, {}, duration=0.1)
        data = json.loads((self.directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertFalse(data["control_restored"])
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No stop after mode-handover displacement")
            time.sleep(0.01)

    def test_record_failure_preserves_failed_file_and_attempts_control_exit(self):
        original = self.robot.get_leader_joint_angles
        count = [0]
        def break_after_start():
            count[0] += 1
            if count[0] == 4:
                raise OSError("test record failure")
            return original()
        with patch.object(self.robot, "get_leader_joint_angles", side_effect=break_after_start):
            with self.assertRaises(RuntimeError):
                record_hand(self.robot, self.directory, {}, duration=0.3)
        data = json.loads((self.directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertTrue(data["control_restored"])
        self.assertTrue(data["samples"])
        with self.assertRaises(ValueError):
            load_teaching(self.directory, self.robot.get_config()["joint_limits"])

    def test_replay_and_reverse_home_follow_joint_waypoints(self):
        points = [s["q_rad"] for s in (sample(0, 0), sample(0.1, 0.6), sample(0.2, 1.2))]
        run = execute_path(self.robot, points, output_dir=Path(self.temp.name) / "runs")
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["completed_segments"]), 3)
        self.assertIsNone(summary["max_endpoint_error_mm"])
        self.assertLess(summary["max_joint_error_deg"], 0.01)
        execute_path(self.robot, points, reverse=True, output_dir=Path(self.temp.name) / "runs")
        values = [int.from_bytes(f.data[:4], "big", signed=True)
                  for f in self.device.host_frames if f.arbitration_id == 0x155]
        self.assertEqual(values, [0, 600, 1200, 1200, 600, 0])
        self.assertFalse(any(f.arbitration_id == 0x471 for f in self.device.host_frames))

    def test_mismatched_start_never_sends_a_bridge_target(self):
        points = [s["q_rad"] for s in (sample(0, 3), sample(0.1, 4))]
        with self.assertRaises(RuntimeError):
            execute_path(self.robot, points, output_dir=self.temp.name)
        self.assertFalse(self.device.host_frames)

    def test_all_seven_joints_including_negative_angles_are_replayed(self):
        target = [math.radians(v) for v in (-0.6, 0.5, -0.4, 0.2, -0.3, 0.2, -0.7)]
        execute_path(self.robot, [[0.0] * 7, target], output_dir=self.temp.name)
        actual = list(self.robot.get_joint_angles().msg)
        for measured, expected in zip(actual, target):
            self.assertLess(abs(measured - expected), math.radians(0.002))

    def test_replay_cycles_and_final_return_use_reverse_route(self):
        points = [s["q_rad"] for s in (sample(0, 0), sample(0.1, 0.6))]
        execute_path(self.robot, points, cycles=2, return_to_start=True, output_dir=self.temp.name)
        values = [int.from_bytes(f.data[:4], "big", signed=True)
                  for f in self.device.host_frames if f.arbitration_id == 0x155]
        self.assertEqual(values, [0, 600, 600, 0, 0, 600, 600, 0])

    def test_recording_state_blocks_disabled_motors_before_leader_mode(self):
        with self.device._state_lock:
            self.device._joints_enabled[0] = False
        time.sleep(0.08)
        with self.assertRaises(RuntimeError):
            record_hand(self.robot, self.directory, {}, duration=0.1)
        self.assertFalse(self.device.host_frames)
        self.assertFalse(self.directory.exists())

    def test_mit_state_blocks_record_with_actual_mode_and_without_teaching_request(self):
        with self.device._state_lock:
            self.device._mode_feedback = 6
        time.sleep(0.08)
        with self.assertRaisesRegex(RuntimeError, "MOVE_MIT.*값 6"):
            record_hand(self.robot, self.directory, {}, duration=0.1)
        self.assertFalse(self.device.host_frames)
        self.assertFalse(self.directory.exists())

    def test_prepare_changes_disabled_mit_to_joint_mode_without_targets_or_enable(self):
        with self.device._state_lock:
            self.device._mode_feedback = 6
            self.device._joints_enabled = [False] * 7
            self.device.reported_arm_status = 6
        time.sleep(0.08)
        prepare_joint_mode(self.robot)
        self.assertEqual(int(self.robot.get_arm_status().msg.mode_feedback), 1)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        frames = self.device.host_frames
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].arbitration_id, 0x151)
        self.assertEqual(bytes(frames[0].data[:4]), bytes([1, 1, 5, 0]))

    def test_prepare_rejects_enabled_or_partial_motors_without_any_command(self):
        for enabled in ([True] * 7, [False] * 6 + [True]):
            with self.subTest(enabled=enabled):
                with self.device._state_lock:
                    self.device._mode_feedback = 6
                    self.device._joints_enabled = list(enabled)
                time.sleep(0.08)
                with self.assertRaisesRegex(RuntimeError, "모터 7개가 모두 비활성"):
                    prepare_joint_mode(self.robot)
                self.assertFalse(self.device.host_frames)

    def test_prepare_requires_joint_mode_ack_not_just_can_control(self):
        with self.device._state_lock:
            self.device._mode_feedback = 6
            self.device._joints_enabled = [False] * 7
            self.device.ignore_motion_change = True
        time.sleep(0.08)
        with self.assertRaisesRegex(FeedbackUnavailableError, "mode_feedback=MOVE_MIT"):
            prepare_joint_mode(self.robot, timeout=0.2)
        self.assertTrue(all(f.arbitration_id == 0x151 for f in self.device.host_frames))

    def test_control_failure_persists_mit_state_and_reason_for_direct_inspection(self):
        with self.device._state_lock:
            self.device._mode_feedback = 6
        time.sleep(0.08)
        with patch("nero_teach.connect_known_nero", return_value=(self.robot, "known", "v112")):
            with patch("sys.argv", ["nero_teach.py", "control", "--firmware", "v112", "--execute"]):
                self.assertEqual(main(), 1)
        logs = list(self.log_root.glob("*.jsonl"))
        self.assertEqual(len(logs), 1)
        events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["event"], "failed")
        self.assertEqual(events[-1]["mode_feedback_value"], 6)
        self.assertTrue(events[-1]["status_fresh"])
        self.assertIn("이동 모드", events[-1]["error"])
        self.assertTrue(any(e["event"] == "after_follower_feedback" for e in events))
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_control_command_exits_leader_mode_with_stale_regular_feedback(self):
        self.robot.set_leader_mode()
        wait_for_leader(self.robot, newer_than=feedback_time())
        time.sleep(0.6)
        with patch("nero_teach.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_teach.py", "control", "--execute"]):
                self.assertEqual(main(), 0)
        self.assertFalse(self.device.leader)

    def test_rejected_entry_keeps_normal_feedback_and_still_restores_control(self):
        self.device.ignore_leader_request = True
        with self.assertRaises(RuntimeError):
            record_hand(self.robot, self.directory, {}, duration=0.1)
        data = json.loads((self.directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertTrue(data["control_restored"])
        self.assertEqual(data["samples"], [])
        self.assertTrue(self.device.normal_push)
        modes = [f.data[0] for f in self.device.host_frames if f.arbitration_id == 0x470]
        self.assertEqual(modes, [0xFA, 0xFC])
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_control_recovers_silent_stiff_arm_with_no_leader_packets(self):
        # Reproduce the old failure: normal push disabled, leader entry ignored.
        self.device.normal_push = False
        time.sleep(0.6)
        self.assertIsNone(self.robot.get_leader_joint_angles())
        with patch("nero_teach.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_teach.py", "control", "--execute"]):
                self.assertEqual(main(), 0)
        self.assertTrue(self.device.normal_push)
        modes = [f.data[0] for f in self.device.host_frames if f.arbitration_id == 0x470]
        self.assertEqual(modes, [0xFC])
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_reported_mode6_waits_for_can_takeover_and_preserves_move_l(self):
        with self.device._state_lock:
            self.device.leader = True
            self.device.emit_leader_packets = False
            self.device._ctrl_mode = 6
            self.device._mode_feedback = 2  # Actual user feedback: MOVE_L.
            self.device.reclaim_delay = 0.2
        time.sleep(0.08)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 6)
        started = time.monotonic()
        restore_control(self.robot)
        self.assertGreaterEqual(time.monotonic() - started, 0.7)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        takeover = [f for f in self.device.host_frames if f.arbitration_id == 0x151 and f.data[1] != 255]
        self.assertEqual(len(takeover), 1)
        self.assertEqual(bytes(takeover[0].data[:4]), bytes([1, 2, 5, 0]))
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_mode6_without_can_ack_times_out_without_any_target(self):
        with self.device._state_lock:
            self.device.leader = True
            self.device.emit_leader_packets = False
            self.device._ctrl_mode = 6
            self.device.ignore_can_takeover = True
        time.sleep(0.08)
        with self.assertRaisesRegex(FeedbackUnavailableError, "CAN 제어 복귀 시간 초과"):
            restore_control(self.robot, timeout=0.2)
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_disabled_mode6_recovers_without_releasing_brakes_or_enabling(self):
        with self.device._state_lock:
            self.device.leader = True
            self.device.emit_leader_packets = False
            self.device._ctrl_mode = 6
            self.device._mode_feedback = 2
            self.device._joints_enabled = [False] * 7
            self.device.reported_arm_status = 6
        time.sleep(0.08)
        restore_control(self.robot)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), 6)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        self.assertFalse(any(f.arbitration_id in (0x150, 0x152, 0x153, 0x154,
                                                0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_partial_enable_state_blocks_control_role_change(self):
        with self.device._state_lock:
            self.device._ctrl_mode = 6
            self.device._joints_enabled[0] = False
        time.sleep(0.08)
        with self.assertRaisesRegex(RuntimeError, "모터/팔 상태가 일관되지"):
            restore_control(self.robot)
        self.assertFalse(any(f.arbitration_id in (0x470, 0x471, 0x155, 0x156, 0x157, 0x170)
                             for f in self.device.host_frames))

    def test_record_uses_normal_packets_only_when_mode6_is_confirmed(self):
        self.device.emit_leader_packets = False
        self.device.hand_motion = True
        samples = record_hand(self.robot, self.directory, {}, duration=0.25)
        self.assertGreater(len(samples), 4)
        self.assertGreater(samples[-1]["q_rad"][0], samples[0]["q_rad"][0])
        _, data = load_teaching(self.directory, self.robot.get_config()["joint_limits"])
        self.assertEqual(data["source"], "normal_joint_packets_in_linkage_mode")
        self.assertTrue(data["control_restored"])
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_control_refuses_role_change_if_no_feedback_can_be_recovered(self):
        self.device.normal_push = False
        self.device.reject_push_enable = True
        time.sleep(0.6)
        with self.assertRaises(FeedbackUnavailableError):
            restore_control(self.robot)
        # Only a reporting request was sent; no unobserved role or target change.
        self.assertFalse(any(f.arbitration_id in (0x470, 0x155, 0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_known_firmware_recovers_without_any_firmware_query(self):
        self.device.normal_push = False
        time.sleep(0.6)
        with patch("nero_teach.connect_nero", side_effect=AssertionError("Do not query firmware")):
            with patch("nero_teach.AgxArmFactory.create_arm", return_value=self.robot) as factory:
                with patch.object(self.robot, "get_firmware", side_effect=AssertionError("No firmware query")):
                    with patch("sys.argv", ["nero_teach.py", "control", "--firmware", "v112", "--execute"]):
                        self.assertEqual(main(), 0)
        self.assertTrue(self.device.normal_push)
        self.assertEqual(factory.call_args[0][0]["firmeware_version"], "v112")
        self.assertFalse(any(f.arbitration_id in (0x155, 0x156, 0x157, 0x170, 0x471, 0x4AF)
                             for f in self.device.host_frames))

    def test_timeout_stops_and_keeps_partial_replay_log(self):
        self.device.accept_joints = False
        points = [s["q_rad"] for s in (sample(0, 0), sample(0.1, 1))]
        with self.assertRaises(RuntimeError):
            execute_path(self.robot, points, timeout=0.15, return_to_start=True, output_dir=self.temp.name)
        summaries = list(Path(self.temp.name).glob("*/summary.json"))
        self.assertEqual(len(summaries), 1)
        self.assertEqual(json.loads(summaries[0].read_text(encoding="utf-8"))["status"], "failed")
        targets = [int.from_bytes(f.data[:4], "big", signed=True)
                   for f in self.device.host_frames if f.arbitration_id == 0x155]
        self.assertEqual(targets, [0, 1000])  # No automatic return after failure.
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No stop after timeout")
            time.sleep(0.01)

    def test_joint_observer_failure_requests_stop_after_transmission(self):
        calls = [0]
        def observer():
            calls[0] += 1
            if calls[0] >= 2:
                raise OSError("test logger failure")
        with self.assertRaises(OSError):
            move_joint_target(self.robot, sample(0, 1)["q_rad"], on_feedback=observer)
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No stop after logger failure")
            time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
