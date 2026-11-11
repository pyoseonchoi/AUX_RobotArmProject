"""Emergency-stop recovery checks; all command tests use virtual CAN."""

import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nero_reset import main, reset_emergency_stop, reset_feedback
from tests.test_nero_enable import disabled_robot
from tests.test_nero_move_to import CartesianSlave, VirtualRobotCase


class ResetPrecheckTests(unittest.TestCase):
    def test_requires_emergency_state(self):
        with self.assertRaises(RuntimeError):
            reset_feedback(disabled_robot(arm_status=0, enabled=[True] * 7))

    def test_requires_all_motors_enabled(self):
        with self.assertRaises(RuntimeError):
            reset_feedback(disabled_robot(arm_status=1, enabled=[True, False] + [True] * 5))

    def test_rejects_stale_driver_and_driver_fault(self):
        robot = disabled_robot(arm_status=1, enabled=[True] * 7)
        driver = robot.get_driver_states(1)
        driver.timestamp -= 2
        with self.assertRaises(RuntimeError):
            reset_feedback(robot)
        driver.timestamp = time.time()
        driver.msg.foc_status.driver_overcurrent = True
        with self.assertRaises(RuntimeError):
            reset_feedback(robot)

    def test_rejects_out_of_range_pose(self):
        robot = disabled_robot(arm_status=1, enabled=[True] * 7)
        robot.get_joint_angles = lambda: SimpleNamespace(msg=[2.0] + [0.0] * 6, timestamp=time.time())
        with self.assertRaises(RuntimeError):
            reset_feedback(robot)


class EmergencySlave(CartesianSlave):
    def __init__(self, channel):
        self.emergency = True
        self.ignore_reset = False
        super().__init__(channel)

    def _apply_host_command(self, aid, data):
        super()._apply_host_command(aid, data)
        if aid == 0x150 and data[0] == 2 and not self.ignore_reset:
            with self._state_lock:
                self.emergency = False
                self._joints_enabled = [False] * 7
                self._proactive_refresh = True

    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        with self._state_lock:
            arm_status = 1 if self.emergency else (0 if all(self._joints_enabled) else 6)
        for frame in frames:
            if frame.arbitration_id == 0x2A1:
                frame.data[1] = arm_status
                frame.data[4] = 1
        return frames


class VirtualResetTests(VirtualRobotCase):
    slave_type = EmergencySlave

    def test_default_is_read_only(self):
        with patch("nero_reset.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_reset.py"]):
                self.assertEqual(main(), 0)
        self.assertFalse(self.device.host_frames)

    def test_execute_sends_one_reset_without_enable_or_move(self):
        with patch("nero_reset.connect_nero", return_value=(self.robot, "1.12", "v112")):
            with patch("sys.argv", ["nero_reset.py", "--execute"]):
                self.assertEqual(main(), 0)
        frames = self.device.host_frames
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].arbitration_id, 0x150)
        self.assertEqual(bytes(frames[0].data), bytes([2] + [0] * 7))
        self.assertFalse(any(self.robot.get_joints_enable_status_list()))
        self.assertEqual(self.robot.get_arm_status().msg.arm_status, 6)

    def test_ignored_reset_is_not_retried(self):
        self.device.ignore_reset = True
        with self.assertRaises(RuntimeError):
            reset_emergency_stop(self.robot, timeout=0.1)
        self.assertEqual(len(self.device.host_frames), 1)
        self.assertEqual(self.device.host_frames[0].data[0], 2)

    def test_changing_joint_angles_prevent_reset(self):
        original = self.robot.get_joint_angles
        started = time.monotonic()
        def moving():
            values = list(original().msg)
            values[0] += math.radians(time.monotonic() - started)
            return SimpleNamespace(msg=values, timestamp=time.time())
        with patch.object(self.robot, "get_joint_angles", side_effect=moving):
            with self.assertRaises(RuntimeError):
                reset_emergency_stop(self.robot)
        self.assertFalse(self.device.host_frames)


if __name__ == "__main__":
    unittest.main()
