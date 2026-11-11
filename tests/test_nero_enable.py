"""Regression checks for enabling a disabled/braked arm with an unreached target."""

import contextlib
import io
import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from tests.slaves.nero_can_slave import NeroCanSlave
from nero_enable import DRIVER_ERROR_FIELDS, check_before_enable, main
from nero_move_joint import check_before_move


def feedback(msg, age=0):
    return SimpleNamespace(msg=msg, timestamp=time.time() - age)


def disabled_robot(arm_status=6, motion_status=1, enabled=None):
    enabled = enabled or [False] * 7
    drivers = []
    for flag in enabled:
        foc = SimpleNamespace(**{name: False for name in DRIVER_ERROR_FIELDS}, driver_enable_status=flag)
        drivers.append(feedback(SimpleNamespace(foc_status=foc)))
    return SimpleNamespace(
        get_joint_angles=lambda: feedback([0.0] * 7),
        get_arm_status=lambda: feedback(SimpleNamespace(arm_status=arm_status, motion_status=motion_status, ctrl_mode=1)),
        get_config=lambda: {"joint_limits": {f"joint{i}": [-1, 1] for i in range(1, 8)}},
        get_driver_states=lambda index: drivers[index - 1],
    )


class EnablePrecheckTests(unittest.TestCase):
    def test_enable_accepts_disabled_braked_unreached_state(self):
        self.assertTrue(check_before_enable(disabled_robot(), allow_disabled_not_reached=True))

    def test_default_precheck_still_blocks_unreached_target(self):
        with self.assertRaises(RuntimeError):
            check_before_enable(disabled_robot())

    def test_enable_accepts_already_enabled_normal_can_state(self):
        robot = disabled_robot(arm_status=0, enabled=[True] * 7)
        self.assertFalse(check_before_enable(robot, allow_enabled_not_reached=True))

    def test_allowance_does_not_apply_to_enabled_or_partial_arm(self):
        for enabled in ([True] * 7, [True] + [False] * 6):
            with self.subTest(enabled=enabled), self.assertRaises(RuntimeError):
                check_before_enable(disabled_robot(enabled=enabled), allow_disabled_not_reached=True)

    def test_errors_and_unknown_motion_states_are_not_bypassed(self):
        for arm, motion in ((1, 1), (7, 1), (6, 2), (0, 1)):
            with self.subTest(arm=arm, motion=motion), self.assertRaises(RuntimeError):
                check_before_enable(disabled_robot(arm, motion), allow_disabled_not_reached=True)

    def test_emergency_stop_message_identifies_recovery_instead_of_enable(self):
        with self.assertRaises(RuntimeError) as raised:
            check_before_enable(disabled_robot(arm_status=1, enabled=[True] * 7))
        self.assertIn("EMERGENCY_STOP", str(raised.exception))
        self.assertIn("nero_reset.py", str(raised.exception))

    def test_stale_feedback_and_driver_fault_are_not_bypassed(self):
        robot = disabled_robot()
        driver = robot.get_driver_states(1)
        driver.timestamp -= 2
        with self.assertRaises(RuntimeError):
            check_before_enable(robot, allow_disabled_not_reached=True)
        driver.timestamp = time.time()
        driver.msg.foc_status.driver_overcurrent = True
        with self.assertRaises(RuntimeError):
            check_before_enable(robot, allow_disabled_not_reached=True)


class DisabledTargetSlave(NeroCanSlave):
    def _master_feedback_messages(self):
        frames = super()._master_feedback_messages()
        with self._state_lock:
            disabled = not any(self._joints_enabled)
        for frame in frames:
            if frame.arbitration_id == 0x2A1:
                frame.data[1] = 6 if disabled else 0
                frame.data[4] = 1  # Preserve unreached status even after enable.
        return frames


class VirtualEnableRegressionTests(unittest.TestCase):
    def test_reported_state_enables_without_reset_or_target_frames(self):
        channel = "disabled_target_test_" + uuid.uuid4().hex
        device = DisabledTargetSlave(channel)
        device.start()
        robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.V112,
            interface="virtual", channel=channel,
        ))
        stop = threading.Event()
        def feed():
            while not stop.wait(0.02):
                device.emit_proactive_feedback_once()
        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        try:
            robot.connect()
            deadline = time.monotonic() + 2
            while robot.get_arm_status() is None or robot.get_driver_states(7) is None:
                if time.monotonic() > deadline:
                    self.fail("Virtual feedback startup timed out")
                time.sleep(0.02)
            state = robot.get_arm_status().msg
            self.assertEqual(state.arm_status, 6)
            self.assertEqual(state.motion_status, 1)
            self.assertFalse(any(robot.get_joints_enable_status_list()))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with patch("nero_enable.connect_nero", return_value=(robot, "1.12", "v112")):
                    with patch("sys.argv", ["nero_enable.py"]):
                        self.assertEqual(main(), 0)
            self.assertTrue(all(robot.get_joints_enable_status_list()))
            self.assertEqual(robot.get_arm_status().msg.arm_status, 0)
            self.assertEqual(robot.get_arm_status().msg.motion_status, 1)
            # With CAN already selected, enable must send only enable packets.
            frames = device.host_frames
            self.assertTrue(frames)
            self.assertTrue(all(frame.arbitration_id == 0x471 and frame.data[1] == 2 for frame in frames))
            robot.connect()
            device.emit_proactive_feedback_once()
            time.sleep(0.05)
            before = len(device.host_frames)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(check_before_move(robot)), 7)
            time.sleep(0.05)
            self.assertEqual(len(device.host_frames), before)
            # Re-running enable in this state is read-only and idempotent.
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("nero_enable.connect_nero", return_value=(robot, "1.12", "v112")):
                    with patch("sys.argv", ["nero_enable.py"]):
                        self.assertEqual(main(), 0)
            self.assertEqual(len(device.host_frames), before)
        finally:
            stop.set()
            feeder.join(timeout=1)
            robot.disconnect()
            device.stop()


if __name__ == "__main__":
    unittest.main()
