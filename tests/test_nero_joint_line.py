"""Joint moves led along the straight joint-space line, so all joints arrive together."""

import math
import unittest

from nero_move_joint import JointLine, move_joint_target
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_stop_policy import PolicySlave


def rad(*degrees):
    return [math.radians(v) for v in degrees] + [0.0] * (7 - len(degrees))


def deg(values):
    return [math.degrees(v) for v in values]


class JointLineTests(unittest.TestCase):
    def test_target_stays_one_lead_ahead_of_the_measured_progress(self):
        line = JointLine(rad(0, 0), rad(10, 2), lead_degrees=2)
        first = deg(line.advance(rad(0, 0)))
        self.assertAlmostEqual(first[0], 2.0)
        self.assertAlmostEqual(first[1], 0.4)   # Same fraction of its own travel.
        self.assertIsNone(line.advance(rad(0, 0)))          # No progress: nothing new to send.
        second = deg(line.advance(rad(1, 0.4)))
        self.assertAlmostEqual(second[0], 3.0)
        self.assertAlmostEqual(second[1], 0.6)
        self.assertIsNone(line.advance(rad(0.5, 0.4)))      # Never moves the target back.
        self.assertEqual(line.advance(rad(8.5, 1.9)), rad(10, 2))  # Ends exactly on the goal.
        self.assertIsNone(line.advance(rad(10, 2)))

    def test_slowest_of_the_long_travel_joints_sets_the_pace(self):
        line = JointLine(rad(0, 0, 0), rad(10, -8, 0.5), lead_degrees=2)
        line.advance(rad(0, 0, 0))
        # J1 is half way but J2 only a quarter; J3 travels too little to pace anything.
        ahead = deg(line.advance(rad(5, -2, 0.5)))
        self.assertAlmostEqual(ahead[0], 4.5)
        self.assertAlmostEqual(ahead[1], -3.6)

    def test_short_move_gets_the_goal_at_once(self):
        line = JointLine(rad(1, 1), rad(2, 1.5), lead_degrees=2)
        self.assertEqual(line.advance(rad(1, 1)), rad(2, 1.5))
        self.assertIsNone(line.advance(rad(1, 1)))


class JointLineVirtualTests(VirtualRobotCase):
    slave_type = PolicySlave

    def follow(self, **options):
        """Move J1 by 20 and J2 by 4 degrees; return how far J2 strays from J1's fraction."""
        self.device.ramp_deg = 0.2  # Every joint turns at the same rate, like firmware 1.12.
        worst = [0.0]
        def watch():
            q = self.robot.get_joint_angles().msg
            worst[0] = max(worst[0], abs(math.degrees(q[1]) - 0.2 * math.degrees(q[0])))
        actual = move_joint_target(self.robot, rad(20, 4), timeout=5, max_step_degrees=25,
                                   allow_long_move=True, on_feedback=watch, **options)
        self.assertAlmostEqual(math.degrees(actual[0]), 20, places=2)
        self.assertAlmostEqual(math.degrees(actual[1]), 4, places=2)
        return worst[0]

    def sent_j1(self):
        return [int.from_bytes(f.data[:4], "big", signed=True) / 1000
                for f in self.device.host_frames if f.arbitration_id == 0x155]

    def test_single_target_lets_the_short_joint_run_ahead(self):
        self.assertGreater(self.follow(), 2.0)
        self.assertEqual(self.sent_j1(), [20.0])

    def test_led_joints_keep_the_same_fraction_and_arrive_together(self):
        self.assertLess(self.follow(lead_degrees=2), 0.6)
        sent = self.sent_j1()
        self.assertGreater(len(sent), 5)
        self.assertEqual(sent, sorted(sent))
        self.assertAlmostEqual(sent[0], 2.0, places=2)
        self.assertEqual(sent[-1], 20.0)
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))

    def test_arm_that_does_not_follow_is_never_commanded_past_the_lead(self):
        self.device.ramp_deg = 0.0  # Targets are received but nothing moves.
        with self.assertRaisesRegex(RuntimeError, "시간 초과"):
            move_joint_target(self.robot, rad(20, 4), timeout=0.3, max_step_degrees=25,
                              allow_long_move=True, lead_degrees=2)
        self.assertEqual(self.sent_j1(), [2.0])

    def test_lead_must_be_reasonable(self):
        for lead in (0, 0.2, 11, float("nan")):
            with self.subTest(lead=lead), self.assertRaises(ValueError):
                move_joint_target(self.robot, rad(1), lead_degrees=lead)
        for rate in (0, 5, 500, float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                move_joint_target(self.robot, rad(1), lead_degrees=2, update_hz=rate)
        self.assertFalse(self.device.host_frames)

    def test_slower_refresh_still_arrives_together(self):
        self.assertLess(self.follow(lead_degrees=2, update_hz=20), 0.6)
        self.assertEqual(self.sent_j1()[-1], 20.0)


if __name__ == "__main__":
    unittest.main()
