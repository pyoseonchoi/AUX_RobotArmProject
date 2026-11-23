"""Continuous joint-path following (used for teaching replay), offline and on virtual CAN."""

import math
import time
import unittest

from nero_move_joint import JointPath, follow_joint_path
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_stop_policy import PolicySlave


def rad(*degrees):
    return [math.radians(v) for v in degrees] + [0.0] * (7 - len(degrees))


def deg(values):
    return [round(math.degrees(v), 6) for v in values]


# An L-shaped path: J1 goes to 10 degrees, then J2 goes to 10 degrees.
BEND = [rad(0, 0), rad(5, 0), rad(10, 0), rad(10, 5), rad(10, 10)]


class JointPathTests(unittest.TestCase):
    def test_length_is_measured_by_the_fastest_joint(self):
        path = JointPath([rad(0, 0), rad(4, 1), rad(4, 7)], lead_degrees=2)
        self.assertAlmostEqual(math.degrees(path.total), 4 + 6)

    def test_target_leads_the_arm_along_the_path_and_turns_the_corner(self):
        path = JointPath(BEND, lead_degrees=2)
        target, distance = path.advance(rad(0, 0))
        self.assertEqual(deg(target)[:2], [2.0, 0.0])
        self.assertAlmostEqual(distance, 0.0)
        target, _ = path.advance(rad(9, 0))
        self.assertEqual(deg(target)[:2], [10.0, 1.0])  # 1 degree of J1 left, then 1 of J2.
        self.assertIsNone(path.advance(rad(9, 0))[0])   # No new target without progress.

    def test_progress_never_moves_back_and_ends_on_the_last_point(self):
        path = JointPath(BEND, lead_degrees=2)
        path.advance(rad(10, 4))
        before = path.progress
        path.advance(rad(9, 0))
        self.assertEqual(path.progress, before)
        target, _ = path.advance(rad(10, 9))
        self.assertEqual(target, BEND[-1])
        self.assertTrue(path.finished)

    def test_distance_from_the_path_is_reported(self):
        path = JointPath(BEND, lead_degrees=2)
        _, distance = path.advance(rad(5, 0, 3))
        self.assertAlmostEqual(math.degrees(distance), 3.0)


class FollowJointPathTests(VirtualRobotCase):
    slave_type = PolicySlave

    def setUp(self):
        super().setUp()
        self.device.ramp_deg = 0.2  # Every joint turns at the same rate, like firmware 1.12.

    def j_targets(self, index):
        frame_id, offset = ((0x155, 0), (0x155, 4))[index]
        return [int.from_bytes(f.data[offset:offset + 4], "big", signed=True) / 1000
                for f in self.device.host_frames if f.arbitration_id == frame_id]

    def test_follows_the_whole_path_in_one_continuous_motion(self):
        watched = []
        actual, deviation = follow_joint_path(
            self.robot, BEND, timeout=5, lead_degrees=2,
            on_feedback=lambda: watched.append(list(self.robot.get_joint_angles().msg)))
        self.assertLess(max(abs(a - b) for a, b in zip(actual, BEND[-1])), math.radians(0.1))
        self.assertLess(deviation, 2.5)
        # Many small targets, never moving back, instead of one stop per waypoint.
        j1, j2 = self.j_targets(0), self.j_targets(1)
        self.assertGreater(len(j1), len(BEND))
        self.assertEqual(j1, sorted(j1))
        self.assertEqual(j2, sorted(j2))
        # The corner is kept: J2 only moves once J1 is (almost) done.
        for q in watched:
            if math.degrees(q[1]) > 1.0:
                self.assertGreater(math.degrees(q[0]), 8.0)
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))

    def test_start_away_from_the_path_sends_nothing(self):
        with self.assertRaisesRegex(RuntimeError, "시작점"):
            follow_joint_path(self.robot, [rad(3, 0), rad(6, 0)], timeout=2)
        self.assertFalse(self.device.host_frames)

    def test_arm_leaving_the_path_requests_stop(self):
        # J2 should stay at 0 but is pushed to 3 degrees while J1 follows the path.
        original = self.device._apply_host_command

        def push_j2(aid, data):
            original(aid, data)
            if aid == 0x155 and self.device.goal is not None:
                self.device.goal[1] = math.radians(3)
        self.device._apply_host_command = push_j2
        with self.assertRaisesRegex(RuntimeError, "벗어났습니다"):
            follow_joint_path(self.robot, [rad(0, 0), rad(8, 0)], timeout=5, lead_degrees=2,
                              max_deviation_degrees=1.0)
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() > deadline:
                self.fail("No stop after leaving the path")
            time.sleep(0.01)

    def test_arm_that_stops_following_times_out_and_requests_stop(self):
        self.device.ramp_deg = 0.0  # Targets are received but nothing moves.
        with self.assertRaisesRegex(RuntimeError, "진행 없음"):
            follow_joint_path(self.robot, BEND, timeout=0.5, lead_degrees=2)
        j1 = self.j_targets(0)
        self.assertLessEqual(max(j1), 2.0)  # Never commanded past one lead ahead of the arm.
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() > deadline:
                self.fail("No stop after the timeout")
            time.sleep(0.01)

    def test_invalid_options_and_points_are_rejected(self):
        for kwargs in (dict(lead_degrees=0.1), dict(update_hz=5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                follow_joint_path(self.robot, BEND, **kwargs)
        with self.assertRaises(ValueError):
            follow_joint_path(self.robot, [rad(0)])
        with self.assertRaisesRegex(RuntimeError, "범위"):
            follow_joint_path(self.robot, [rad(0), rad(170)])
        self.assertFalse(self.device.host_frames)


if __name__ == "__main__":
    unittest.main()
