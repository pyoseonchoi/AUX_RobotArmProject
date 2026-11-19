"""Motor power requests with out-of-range feedback; virtual CAN only."""

import time
from unittest.mock import patch

from nero_disable import disable_only, main as disable_main
from nero_enable import main as enable_main
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_teach import TeachingSlave


class RangeSlave(TeachingSlave):
    def __init__(self, channel):
        self.ignore_disable = False
        super().__init__(channel)
        self.reported_arm_status = 4  # TARGET_POS_EXCEEDS_LIMIT
        self.current_joints[0] = 3.0  # Beyond this SDK's J1 upper limit.

    def _apply_host_command(self, aid, data):
        if self.ignore_disable and aid == 0x471 and data[1] == 1:
            return
        super()._apply_host_command(aid, data)

    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        for frame in frames:
            if frame.arbitration_id == 0x2A1:
                frame.data[4] = 1  # The old disable guard blocked this.
        return frames


class MotorPowerTests(VirtualRobotCase):
    slave_type = RangeSlave

    def assert_only_power(self, flag):
        time.sleep(0.04)
        self.assertTrue(self.device.host_frames)
        self.assertTrue(all(f.arbitration_id == 0x471 and f.data[1] == flag
                            for f in self.device.host_frames))

    def test_disable_cli_with_out_of_range_angles_and_unreached_target(self):
        self.assertGreater(self.robot.get_joint_angles().msg[0],
                           self.robot.get_config()["joint_limits"]["joint1"][1])
        self.assertEqual(int(self.robot.get_arm_status().msg.motion_status), 1)
        with patch("nero_disable.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_disable.py"]):
            self.assertEqual(disable_main(), 0)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        self.assert_only_power(1)

    def test_enable_cli_with_out_of_range_angles_and_unreached_target(self):
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        self.assertGreater(self.robot.get_joint_angles().msg[0],
                           self.robot.get_config()["joint_limits"]["joint1"][1])
        with patch("nero_enable.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_enable.py"]):
            self.assertEqual(enable_main(), 0)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [True] * 7)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), 4)
        self.assert_only_power(2)

    def test_disable_does_not_require_status_feedback(self):
        with patch.object(self.robot, "get_arm_status", return_value=None):
            disable_only(self.robot)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        # SDK may also request CAN control when status is absent, but no targets/reset.
        self.assertTrue(all(f.arbitration_id in (0x471, 0x151) for f in self.device.host_frames))

    def test_emergency_stop_does_not_block_disable(self):
        self.device.reported_arm_status = 1
        time.sleep(0.08)
        disable_only(self.robot)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        self.assert_only_power(1)

    def test_ignored_disable_reports_unconfirmed_motors(self):
        self.device.ignore_disable = True
        with self.assertRaisesRegex(RuntimeError, "J1.*J7"):
            disable_only(self.robot, timeout=0.1)
        self.assert_only_power(1)

    def test_cached_disabled_bits_are_not_accepted_without_new_feedback(self):
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        self.assertEqual(self.robot.get_joints_enable_status_list(), [False] * 7)
        self.feed_paused.set()
        with patch.object(self.device, "_send_and_record"):
            time.sleep(0.05)
            with self.assertRaisesRegex(RuntimeError, "비활성화를 확인하지 못했습니다"):
                disable_only(self.robot, timeout=0.15)
        self.assert_only_power(1)
