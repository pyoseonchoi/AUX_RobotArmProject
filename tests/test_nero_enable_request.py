"""Motor enable is independent of prior IK failure; virtual CAN only."""

import time
from unittest.mock import patch

from nero_enable import enable_only, main
from tests.test_nero_move_to import CartesianSlave, VirtualRobotCase


class NoSolutionSlave(CartesianSlave):
    def __init__(self, channel):
        self.arm_status = 2
        self.ignore_enable = False
        super().__init__(channel)

    def _apply_host_command(self, aid, data):
        if self.ignore_enable and aid == 0x471:
            return
        super()._apply_host_command(aid, data)

    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        for frame in frames:
            if frame.arbitration_id == 0x2A1:
                frame.data[1] = self.arm_status
                frame.data[4] = 1
        return frames


class EnableRequestTests(VirtualRobotCase):
    slave_type = NoSolutionSlave

    def disable_fixture(self):
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)

    def test_cli_enables_despite_no_solution_and_reports_remaining_error(self):
        self.disable_fixture()
        with patch("nero_enable.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_enable.py"]):
            self.assertEqual(main(), 0)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [True] * 7)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), 2)
        self.assertTrue(self.device.host_frames)
        self.assertTrue(all(f.arbitration_id == 0x471 and f.data[1] == 2
                            for f in self.device.host_frames))

    def test_already_enabled_no_solution_is_idempotent(self):
        with patch("nero_enable.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_enable.py"]):
            self.assertEqual(main(), 0)
        self.assertFalse(self.device.host_frames)

    def test_ignored_enable_times_out_without_reset_or_stop_or_target(self):
        self.disable_fixture()
        self.device.ignore_enable = True
        with self.assertRaisesRegex(RuntimeError, "활성화를 확인하지 못했습니다"):
            enable_only(self.robot, timeout=0.15)
        self.assertTrue(self.device.host_frames)
        self.assertTrue(all(f.arbitration_id == 0x471 and f.data[1] == 2
                            for f in self.device.host_frames))

    def test_emergency_stop_is_not_bypassed(self):
        self.device.arm_status = 1
        self.disable_fixture()
        with self.assertRaisesRegex(RuntimeError, "EMERGENCY_STOP"):
            enable_only(self.robot)
        self.assertFalse(self.device.host_frames)

    def test_cli_failure_does_not_send_emergency_stop(self):
        self.disable_fixture()
        original = enable_only
        self.device.ignore_enable = True
        with patch("nero_enable.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("nero_enable.enable_only", side_effect=lambda arm: original(arm, timeout=0.15)), \
             patch("sys.argv", ["nero_enable.py"]):
            self.assertEqual(main(), 1)
        self.assertTrue(all(f.arbitration_id == 0x471 for f in self.device.host_frames))

    def test_old_enabled_feedback_cannot_confirm_new_request(self):
        self.feed_paused.set()
        time.sleep(0.1)
        # Stop all feedback after this point, including SDK proactive responses.
        with patch.object(self.device, "_send_and_record"):
            with self.assertRaisesRegex(RuntimeError, "활성화를 확인하지 못했습니다"):
                enable_only(self.robot, timeout=0.15)
