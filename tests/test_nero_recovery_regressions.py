"""Reproduce the Nov 13 hardware log and verify recovery on virtual CAN only."""

import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from nero_diagnostics import SessionDiagnostics
from nero_move_linear import move_linear_target
from nero_move_to import main as move_main
from nero_reset import main as reset_main, reset_emergency_stop, reset_feedback
from nero_teach import load_teaching, record_hand, restore_control
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_reset import EmergencySlave
from tests.test_nero_teach import TeachingSlave


class MITHandoverSlave(TeachingSlave):
    def _apply_host_command(self, aid, data):
        was_teaching = self.leader or self._ctrl_mode == 6
        super()._apply_host_command(aid, data)
        if aid == 0x470 and data[0] == 0xFC and was_teaching:
            with self._state_lock:
                self._mode_feedback = 6


class TeachingHandoverTests(VirtualRobotCase):
    slave_type = MITHandoverSlave

    def test_record_from_move_l_restores_after_follower_reports_mit(self):
        # Actual log: entry L -> fresh leader packets -> follower ctrl6/MIT6.
        self.device._mode_feedback = 2
        self.device.leader_suppresses_normal = True
        self.device.hand_motion = True
        time.sleep(0.08)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "record"
            record_hand(self.robot, directory, {}, duration=0.65)
            _, data = load_teaching(directory, self.robot.get_config()["joint_limits"])
        self.assertTrue(data["control_restored"])
        self.assertEqual(data["pre_teach_motion_mode"], 2)
        self.assertGreater(len(data["samples"]), 10)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertEqual(int(self.robot.get_arm_status().msg.mode_feedback), 2)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [True] * 7)
        self.assertFalse(any(f.arbitration_id in (0x150, 0x152, 0x153, 0x154, 0x155,
                                                0x156, 0x157, 0x170, 0x471)
                             for f in self.device.host_frames))

    def test_standalone_control_recovers_logged_ctrl6_mit6_state(self):
        self.device._ctrl_mode, self.device._mode_feedback = 6, 6
        time.sleep(0.08)
        restore_control(self.robot)
        self.assertEqual(int(self.robot.get_arm_status().msg.ctrl_mode), 1)
        self.assertEqual(int(self.robot.get_arm_status().msg.mode_feedback), 1)
        self.assertTrue(all(f.arbitration_id in (0x151, 0x470) for f in self.device.host_frames))

    def test_can_ack_without_motion_ack_does_not_complete_record(self):
        # Ignore the requested planned mode; CAN rights alone are insufficient.
        self.device.leader_suppresses_normal = True
        self.device.ignore_motion_change = True
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "record"
            with self.assertRaisesRegex(RuntimeError, "CAN 제어 복귀 시간 초과"):
                record_hand(self.robot, directory, {}, duration=0.65)
            data = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertFalse(data["control_restored"])

    def test_move_in_can_mit_refuses_before_any_target_or_speed_setting(self):
        self.device._mode_feedback = 6
        time.sleep(0.08)
        target = list(self.robot.get_flange_pose().msg)
        target[2] -= 0.004
        with self.assertRaisesRegex(RuntimeError, "일반 위치 이동 모드"):
            move_linear_target(self.robot, target)
        self.assertFalse(self.device.host_frames)

    def test_linear_failure_log_includes_target_errors_and_stop(self):
        self.device.accept_motion = False
        with tempfile.TemporaryDirectory() as root:
            logger = SessionDiagnostics(root)
            self.robot._nero_diagnostics = logger
            target = list(self.robot.get_flange_pose().msg)
            target[2] -= 0.004
            try:
                with self.assertRaisesRegex(RuntimeError, "시간 초과"):
                    move_linear_target(self.robot, target, timeout=0.15)
            finally:
                logger.close()
            events = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[0]["event"], "linear_target_sent")
        self.assertEqual(events[0]["target_m_rad"], target)
        self.assertEqual(events[-1]["event"], "emergency_stop_requested")
        self.assertTrue(any(e.get("position_error_mm", 0) >= 3.9 for e in events))

    def test_move_to_passes_default_and_explicit_timeout(self):
        for extra, expected in (([], 30.0), (["--timeout", "60"], 60.0)):
            with patch("nero_move_to.connect_nero", return_value=(self.robot, "1.12", "v112")), \
                 patch("nero_move_to.move_linear_target") as move, \
                 patch("sys.argv", ["nero_move_to.py", "--z", "405"] + extra):
                self.robot.connect()
                self.assertEqual(move_main(), 0)
                self.assertEqual(move.call_args[1]["timeout"], expected)
        self.assertFalse(self.device.host_frames)


class ResetRecoveryTests(VirtualRobotCase):
    slave_type = EmergencySlave

    def test_reset_not_needed_is_noop_when_already_disabled(self):
        self.device.emergency = False
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        with patch("nero_reset.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_reset.py", "--execute"]):
            self.assertEqual(reset_main(), 0)
        self.assertFalse(self.device.host_frames)

    def test_disabled_emergency_is_explained_without_automatic_enable(self):
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        with self.assertRaisesRegex(RuntimeError, "제조사 reset 적용 조건"):
            reset_feedback(self.robot)
        self.assertFalse(self.device.host_frames)

    def test_arm_status_clear_without_motor_disable_is_not_success(self):
        original = self.device._apply_host_command
        def keep_enabled(aid, data):
            original(aid, data)
            if aid == 0x150 and data[0] == 2:
                self.device._joints_enabled = [True] * 7
        with patch.object(self.device, "_apply_host_command", side_effect=keep_enabled):
            with self.assertRaisesRegex(RuntimeError, "모터 비활성"):
                reset_emergency_stop(self.robot, timeout=0.2)
        resets = [f for f in self.device.host_frames if f.arbitration_id == 0x150]
        self.assertEqual(len(resets), 1)
        self.assertFalse(any(f.arbitration_id == 0x471 for f in self.device.host_frames))

    def test_stale_joint_pair_is_not_hidden_by_fresh_joint7(self):
        # Freeze one of the four underlying packet groups.
        from types import SimpleNamespace
        from tests.test_nero_enable import disabled_robot
        robot = disabled_robot(arm_status=1, enabled=[True] * 7)
        robot._parser = SimpleNamespace(**{name: SimpleNamespace(timestamp=time.time())
            for name in ("joint_12", "joint_34", "joint_56", "joint_7")})
        robot._parser.joint_34.timestamp -= 2
        with self.assertRaisesRegex(RuntimeError, "joint_34"):
            reset_feedback(robot)
        self.assertFalse(self.device.host_frames)
