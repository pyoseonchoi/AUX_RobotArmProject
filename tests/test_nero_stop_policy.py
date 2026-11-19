"""When a failed move requests the electronic emergency stop, on virtual CAN.

Rule under test: the stop is skipped only if the controller itself reports
that it refused the current target and the joints are then verified still.
Everything else that fails after a target was sent still requests the stop.
"""

import math
import time
import unittest
from unittest.mock import patch

from nero_move_joint import (ControllerRejected, ProgressTimeout, RejectionWatch, ResendGuard,
                             check_before_move, move_joint_target)
from nero_move_linear import move_linear_target
from nero_read_status import FeedbackUnavailableError
from nero_repeat_air import read_sample
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_teach import TeachingSlave


JOINT_IDS = (0x155, 0x156, 0x157, 0x170)
POSE_IDS = (0x152, 0x153, 0x154)
NO_SOLUTION = 2


class PolicySlave(TeachingSlave):
    """Joint and pose targets are echoed as feedback unless a test says otherwise."""

    def __init__(self, channel):
        self.drop_once = set()        # Target frame IDs to ignore once, like a frame that was not applied.
        self.reject_poses = False     # Answer a pose target with NO_SOLUTION and do not move.
        self.clear_on_target = True   # An accepted target clears an older refusal flag.
        self.drift_after_reject = False
        self.drifting = False
        self.ramp_deg = None          # Degrees per feedback round toward a joint target.
        self.goal = None
        super().__init__(channel)

    def _apply_host_command(self, aid, data):
        if aid in self.drop_once:
            self.drop_once.discard(aid)
            return
        if aid in POSE_IDS and self.reject_poses:
            self.reported_arm_status = NO_SOLUTION
            self.drifting = self.drift_after_reject
            return
        if aid in JOINT_IDS + POSE_IDS and self.clear_on_target and (aid in POSE_IDS or self.accept_joints):
            self.reported_arm_status = 0
        if aid in JOINT_IDS and self.ramp_deg is not None:
            index = {0x155: 0, 0x156: 2, 0x157: 4, 0x170: 6}[aid]
            with self._state_lock:
                self.goal = list(self.goal or self.current_joints)
                for offset in range(1 if index == 6 else 2):
                    value = int.from_bytes(data[4 * offset:4 * offset + 4], "big", signed=True)
                    self.goal[index + offset] = math.radians(value / 1000)
            return
        super()._apply_host_command(aid, data)

    def _master_feedback_messages(self):
        with self._state_lock:
            if self.drifting:
                self.current_joints[0] += math.radians(0.05)
            if self.goal is not None:
                step = math.radians(self.ramp_deg)
                self.current_joints = [c + max(-step, min(step, g - c))
                                       for c, g in zip(self.current_joints, self.goal)]
        return super()._master_feedback_messages()


def joints(degrees):
    return [math.radians(degrees)] + [0.0] * 6


class StopPolicyTests(VirtualRobotCase):
    slave_type = PolicySlave

    def stops(self):
        return [f for f in self.device.host_frames if f.arbitration_id == 0x150 and f.data[0] == 1]

    def wait_for_stop(self):
        deadline = time.monotonic() + 1
        while not self.stops():
            if time.monotonic() >= deadline:
                self.fail("No electronic emergency stop was requested")
            time.sleep(0.01)

    def line_target(self):
        target = list(self.robot.get_flange_pose().msg)
        target[2] -= 0.004
        return target

    def leave_old_refusal(self):
        self.device.reported_arm_status = NO_SOLUTION
        time.sleep(0.1)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), NO_SOLUTION)

    def test_refused_line_with_still_arm_skips_the_stop(self):
        self.device.reject_poses = True
        with self.assertRaisesRegex(ControllerRejected, "NO_SOLUTION"):
            move_linear_target(self.robot, self.line_target(), timeout=5)
        time.sleep(0.1)
        self.assertFalse(self.stops())
        self.assertEqual(self.robot.get_joints_enable_status_list(), [True] * 7)

    def test_refused_line_but_joints_still_moving_requests_the_stop(self):
        self.device.reject_poses = True
        self.device.drift_after_reject = True
        with self.assertRaises(ControllerRejected):
            move_linear_target(self.robot, self.line_target(), timeout=5)
        self.wait_for_stop()

    def test_old_refusal_flag_does_not_block_a_new_move(self):
        self.leave_old_refusal()
        self.assertEqual(len(check_before_move(self.robot)), 7)
        actual = move_joint_target(self.robot, joints(1))
        self.assertAlmostEqual(actual[0], math.radians(1), places=4)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), 0)
        self.assertFalse(self.stops())

    def test_move_completes_even_if_the_old_flag_never_clears(self):
        self.device.clear_on_target = False
        self.leave_old_refusal()
        actual = move_joint_target(self.robot, joints(1))
        self.assertAlmostEqual(actual[0], math.radians(1), places=4)
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), NO_SOLUTION)
        self.assertFalse(self.stops())

    def test_old_flag_and_no_motion_fails_without_the_stop(self):
        self.device.clear_on_target = False
        self.device.accept_joints = False
        self.leave_old_refusal()
        with patch.object(RejectionWatch, "STALL_S", 0.3):
            with self.assertRaisesRegex(ControllerRejected, "남아 있고"):
                move_joint_target(self.robot, joints(1), timeout=5)
        time.sleep(0.1)
        self.assertFalse(self.stops())

    def test_interrupt_under_an_old_flag_still_requests_the_stop(self):
        # The flag predates this target, so it proves nothing about the new move.
        self.device.clear_on_target = False
        self.device.accept_joints = False
        self.leave_old_refusal()
        calls = []
        def interrupt():
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            move_joint_target(self.robot, joints(1), on_feedback=interrupt)
        self.wait_for_stop()

    def test_lost_feedback_requests_the_stop(self):
        calls = []
        def lose_feedback():
            calls.append(1)
            if len(calls) == 2:
                self.device.accept_joints = False
                self.feed_paused.set()
        self.device.accept_joints = False
        with self.assertRaises(FeedbackUnavailableError):
            move_joint_target(self.robot, joints(1), on_feedback=lose_feedback, timeout=5)
        self.wait_for_stop()

    def test_timeout_without_progress_requests_the_stop(self):
        self.device.accept_joints = False
        with self.assertRaisesRegex(RuntimeError, "시간 초과"):
            move_joint_target(self.robot, joints(1), timeout=0.2)
        self.wait_for_stop()

    def test_slow_but_advancing_move_is_not_timed_out(self):
        self.device.ramp_deg = 0.1  # About 5 deg/s: the 2-degree move outlasts the 0.2 s timeout.
        actual = move_joint_target(self.robot, joints(2), timeout=0.2)
        self.assertAlmostEqual(actual[0], math.radians(2), places=4)
        self.assertFalse(self.stops())

    def frames(self, aid):
        return [f for f in self.device.host_frames if f.arbitration_id == aid]

    def test_joint_frame_not_applied_is_resent(self):
        # Replays 2026-11-18 01:28: J7 moved, J1/J2 frame not applied, controller said "reached".
        self.device.drop_once = {0x155}
        target = joints(1)
        target[6] = math.radians(1)
        actual = move_joint_target(self.robot, target, timeout=5)
        self.assertAlmostEqual(actual[0], math.radians(1), places=4)
        self.assertEqual(len(self.frames(0x155)), 2)
        self.assertEqual(len(self.frames(0x170)), 2)  # The whole target is sent again.
        self.assertFalse(self.stops())

    def test_pose_frame_not_applied_is_resent(self):
        self.device.drop_once = {0x153}  # Z/roll part of the pose target.
        target = self.line_target()
        actual = move_linear_target(self.robot, target, timeout=5)
        self.assertAlmostEqual(actual[2], target[2], places=5)
        self.assertEqual(len(self.frames(0x153)), 2)
        self.assertFalse(self.stops())

    def test_resends_are_limited_and_then_the_timeout_applies(self):
        self.device.accept_joints = False
        with self.assertRaisesRegex(RuntimeError, "시간 초과"):
            move_joint_target(self.robot, joints(1), timeout=3)
        self.assertEqual(len(self.frames(0x155)), 1 + ResendGuard.MAX_RESENDS)
        self.wait_for_stop()

    def test_recorder_health_leaves_refusals_to_the_motion_loop(self):
        self.leave_old_refusal()
        self.assertTrue(read_sample(self.robot)[1])
        self.device.reported_arm_status = 5  # A real fault: joint communication error.
        time.sleep(0.1)
        self.assertFalse(read_sample(self.robot)[1])
        with self.assertRaisesRegex(RuntimeError, "정상 CAN 제어 상태가 아닙니다"):
            check_before_move(self.robot)
        self.assertFalse(self.device.host_frames)


class ProgressTimeoutTests(unittest.TestCase):
    def test_only_real_progress_restarts_the_clock(self):
        clock = [100.0]
        with patch("nero_move_joint.time.monotonic", side_effect=lambda: clock[0]):
            waiting = ProgressTimeout(10.0, [1.0, 0.1])
            self.assertFalse(waiting.expired([50.0, 5.0]))
            clock[0] += 9
            self.assertFalse(waiting.expired([48.0, 5.0]))     # First error improved by a full step.
            clock[0] += 9
            self.assertFalse(waiting.expired([48.0, 4.8]))     # Second error improved.
            clock[0] += 9
            self.assertFalse(waiting.expired([47.5, 4.75]))    # Smaller than a step: no restart...
            clock[0] += 1
            self.assertTrue(waiting.expired([47.5, 4.75]))     # ...so the 10 s run out.

    def test_small_improvements_add_up_to_progress(self):
        clock = [0.0]
        with patch("nero_move_joint.time.monotonic", side_effect=lambda: clock[0]):
            waiting = ProgressTimeout(10.0, [1.0])
            waiting.expired([50.0])
            for error in (49.6, 49.2, 48.8):
                clock[0] += 4
                expired = waiting.expired([error])
            self.assertFalse(expired)  # 1.2 in total crossed the step at the third call.


if __name__ == "__main__":
    unittest.main()
