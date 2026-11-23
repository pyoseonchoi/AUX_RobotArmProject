"""Offline kinematics and straight-line planning. No CAN, no robot."""

import math
import unittest

import numpy as np

from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
from nero_kinematics import (TOOL_AXES, PlanError, PlanSettings, arm_angle, arm_angle_gradient,
                             default_limits, forward, from_rotation_vector, height_floor_mm, jacobian,
                             min_singular_value, plan_joint_move, plan_line, pose6, rotation_angle,
                             rotation_vector, rpy_to_matrix, segment_distance_mm, solve_ik, wrist_tilt)


UPRIGHT = [0.0] * 7
BENT = [math.radians(v) for v in (0, 30, 0, 60, 0, 0, 0)]
DOWN = (TOOL_AXES["x"], (0.0, 0.0, -1.0))


def rad(*degrees):
    return np.radians(degrees)


def swivel_change(a, b):
    return abs((arm_angle(a) - arm_angle(b) + math.pi) % (2 * math.pi) - math.pi)


# Joint angles recorded on the real arm during move_l, 2026-11-16 (start, end of the line).
GOOD_LINES = (
    (rad(67.22, 16.21, 6.52, -25.72, -87.74, -10.19, -2.49), rad(37.01, 34.39, 7.99, -51.85, -56.23, -17.82, 4.90)),
    (rad(33.8, 17.4, 5.8, -27.1, 1.8, -4.5, 3.9), rad(21.9, 35.3, 7.7, -49.8, 13.6, -4.1, 9.5)),
)
# Start of the line that ended in NO_SOLUTION at 23:19, and the logged home posture before it.
FAILED_LINE_START = rad(24.7, -10.6, 5.4, 26.3, 9.4, -4.8, -21.5)
HOME_2026_11_16 = rad(37.0, 5.0, 1.6, -4.8, 1.7, -3.2, -6.1)
# 2026-11-17 00:07: a direct 467 mm line from here to (-100, 100, 600) mm drifted 20 mm
# off the line because J1/J5 fell behind (needed 10-15 deg/s, managed about 9).
DRIFT_START = rad(5.2, -88.3, 9.1, 92.8, 15.6, -25.5, -3.6)
DRIFT_TARGET = [-0.10, 0.10, 0.60]


class ControllerModelTests(unittest.TestCase):
    """What firmware 1.12 does during move_l, as far as the recorded runs show."""

    def test_recorded_good_lines_kept_the_arm_angle(self):
        for start, end in GOOD_LINES:
            with self.subTest(start=np.degrees(start).round(1)):
                self.assertLess(math.degrees(swivel_change(start, end)), 0.5)
                # The elbow did swing a lot: the angle is what the controller keeps, not the joints.
                self.assertGreater(np.degrees(np.abs(end - start)).max(), 10)

    def test_failed_line_started_with_the_shoulder_wrist_axis_almost_vertical(self):
        self.assertLess(math.degrees(wrist_tilt(FAILED_LINE_START)), 3.0)
        for start, _ in GOOD_LINES:
            self.assertGreater(math.degrees(wrist_tilt(start)), 4.0)

    def test_arm_angle_gradient_matches_finite_differences(self):
        limits = default_limits()
        random = np.random.default_rng(3)
        for _ in range(5):
            q = random.uniform(limits[:, 0], limits[:, 1]) * 0.7
            numeric = np.zeros(7)
            for i in range(7):
                step = np.zeros(7)
                step[i] = 1e-6
                numeric[i] = ((arm_angle(q + step) - arm_angle(q - step) + math.pi) % (2 * math.pi) - math.pi) / 2e-6
            self.assertLess(np.abs(arm_angle_gradient(q) - numeric).max(), 1e-6)

    def test_ik_can_hold_the_arm_angle(self):
        limits = default_limits()
        start, _ = GOOD_LINES[1]
        position, rotation = forward(start)
        goal = position + [-0.02, 0.0, -0.015]
        q, converged = solve_ik(goal, start, limits, rotation=rotation, swivel=arm_angle(start))
        self.assertTrue(converged)
        self.assertLess(np.linalg.norm(forward(q)[0] - goal), 5e-5)
        self.assertLess(math.degrees(swivel_change(q, start)), 0.02)

    def test_line_cannot_end_with_the_shoulder_wrist_axis_vertical(self):
        with self.assertRaisesRegex(PlanError, "--path joint"):
            plan_line(BENT, [0.0, -0.0235, 0.700])
        plan = plan_joint_move(BENT, [0.0, -0.0235, 0.700])
        self.assertEqual(len(plan.line_joints), 1)

    def test_line_the_joints_could_not_follow_is_not_planned_again(self):
        settings = PlanSettings()
        # The same direct line, under the controller's arm-angle rule, needs far more than the limit.
        _, rotation = forward(DRIFT_START)
        old = plan_line(DRIFT_START, DRIFT_TARGET, settings=PlanSettings(joint_speed_per_percent=100.0))
        self.assertFalse(old.needs_joint_move)
        self.assertGreater(old.peak_joint_speed_deg_s, 2 * settings.joint_speed_limit)
        new = plan_line(DRIFT_START, DRIFT_TARGET)
        self.assertLessEqual(new.peak_joint_speed_deg_s, settings.joint_speed_limit + 1e-9)
        self.assertLess(new.straight_mm, old.straight_mm)

    def test_flange_speed_model_ramps_and_scales_with_speed(self):
        five, ten = PlanSettings(), PlanSettings(speed_percent=10)
        self.assertEqual(five.flange_speed(0, 100), 0)
        self.assertAlmostEqual(five.flange_speed(10, 100), math.sqrt(2 * 40 * 10))
        self.assertAlmostEqual(five.flange_speed(200, 200), 45)
        self.assertAlmostEqual(ten.flange_speed(500, 500), 90)
        self.assertAlmostEqual(ten.joint_speed_limit, 2 * five.joint_speed_limit)
        # The ramp does not scale with the percentage (measured at 5% and 7%).
        self.assertAlmostEqual(ten.flange_speed(10, 500), five.flange_speed(10, 500))

    def test_replanning_the_failed_request_avoids_its_start(self):
        y = forward(HOME_2026_11_16)[0][1]
        plan = plan_line(HOME_2026_11_16, [0.30, y, 0.40])
        self.assertGreaterEqual(plan.min_tilt_deg, PlanSettings().min_tilt_deg)
        self.assertGreater(plan.straight_mm, 0.6 * plan.total_mm)
        self.assertLessEqual(plan.max_joint_step_deg, PlanSettings().max_joint_step_deg)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.limits = default_limits()
        self.random = np.random.default_rng(7)

    def sample(self):
        return self.random.uniform(self.limits[:, 0], self.limits[:, 1])

    def test_forward_matches_sdk_fk(self):
        for _ in range(50):
            q = self.sample()
            position, rotation = forward(q)
            expected = fk_from_mdh(list(get_mdh("nero")), list(q))
            self.assertLess(np.linalg.norm(position - expected[:3]), 1e-12)
            self.assertLess(rotation_angle(rotation, rpy_to_matrix(*expected[3:])), 1e-7)

    def test_pose6_round_trips_including_vertical_pitch(self):
        for q in [self.sample() for _ in range(20)] + [UPRIGHT]:
            position, rotation = forward(q)
            pose = pose6(position, rotation)
            self.assertLess(rotation_angle(rotation, rpy_to_matrix(*pose[3:])), 1e-6)
            self.assertLessEqual(abs(pose[4]), math.pi / 2)

    def test_jacobian_matches_finite_differences(self):
        q, h = self.sample(), 1e-6
        analytic = jacobian(q)
        for i in range(7):
            plus, minus = q.copy(), q.copy()
            plus[i] += h
            minus[i] -= h
            (pa, ra), (pb, rb) = forward(plus), forward(minus)
            self.assertLess(np.abs((pa - pb) / (2 * h) - analytic[:3, i]).max(), 1e-6)
            self.assertLess(np.abs(rotation_vector(ra @ rb.T) / (2 * h) - analytic[3:, i]).max(), 1e-6)

    def test_rotation_vector_round_trips_near_half_turn(self):
        for angle in (1e-10, 0.3, math.pi - 1e-8, math.pi):
            vector = np.array([0.6, 0.0, 0.8]) * angle
            self.assertLess(rotation_angle(from_rotation_vector(vector),
                                           from_rotation_vector(rotation_vector(from_rotation_vector(vector)))), 1e-6)

    def test_upright_is_singular_and_bent_is_not(self):
        self.assertLess(min_singular_value(UPRIGHT), 1e-9)
        self.assertGreater(min_singular_value(BENT), 0.1)

    def test_ik_reaches_full_poses_inside_limits(self):
        goals = []
        while len(goals) < 20:
            goal = self.sample() * 0.6
            if min_singular_value(goal) >= 0.05:  # Damping keeps IK off singular goals by design.
                goals.append(goal)
        for goal in goals:
            position, rotation = forward(goal)
            q, converged = solve_ik(position, goal + 0.2, self.limits, rotation=rotation)
            self.assertTrue(converged)
            reached, orientation = forward(q)
            self.assertLess(np.linalg.norm(reached - position), 5e-5)
            self.assertLess(rotation_angle(orientation, rotation), 5e-4)
            self.assertTrue(np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))

    def test_aim_constrains_only_the_tool_axis(self):
        axis, direction = np.array(DOWN[0], dtype=float), np.array(DOWN[1])
        q, converged = solve_ik([-0.35, 0.0, 0.30], BENT, self.limits, aim=(axis, direction))
        self.assertTrue(converged)
        position, rotation = forward(q)
        self.assertLess(np.linalg.norm(position - [-0.35, 0.0, 0.30]), 5e-5)
        self.assertGreater(float((rotation @ axis) @ direction), math.cos(5e-4))

    def test_unreachable_position_does_not_converge(self):
        _, converged = solve_ik([0.9, 0.0, 0.3], BENT, self.limits)
        self.assertFalse(converged)

    def test_segment_distance(self):
        self.assertAlmostEqual(segment_distance_mm([0.5, 0.01, 0], [0, 0, 0], [1, 0, 0]), 10.0)
        self.assertAlmostEqual(segment_distance_mm([-0.003, 0.004, 0], [0, 0, 0], [1, 0, 0]), 5.0)
        self.assertAlmostEqual(segment_distance_mm([0.003, 0.004, 0], [0, 0, 0], [0, 0, 0]), 5.0)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.limits = default_limits()
        self.settings = PlanSettings()

    def assert_line_is_straight_and_usable(self, plan, fixed_orientation=True):
        start, end = plan.entry_pose[:3], plan.target_pose[:3]
        margin = math.radians(self.settings.limit_margin_deg) - 1e-9
        final = rpy_to_matrix(*plan.target_pose[3:])
        entry = np.array(plan.line_joints[0])
        for q in plan.line_joints:
            q = np.array(q)
            self.assertLess(segment_distance_mm(forward(q)[0], start, end), 0.1)
            # The predicted line is the controller's own: constant arm angle, axis off vertical.
            self.assertLess(math.degrees(swivel_change(q, entry)), 0.05)
            self.assertGreaterEqual(math.degrees(wrist_tilt(q)), self.settings.min_tilt_deg - 1e-9)
            if fixed_orientation:
                self.assertLess(rotation_angle(forward(q)[1], final), 1e-3)
            self.assertTrue(np.all(q >= self.limits[:, 0] + margin) and np.all(q <= self.limits[:, 1] - margin))
            self.assertGreaterEqual(min_singular_value(q), self.settings.min_sigma)
        for a, b in zip(plan.line_joints, plan.line_joints[1:]):
            self.assertLessEqual(np.abs(np.array(a) - np.array(b)).max(),
                                 math.radians(self.settings.max_joint_step_deg) + 1e-9)
        self.assertLess(np.linalg.norm(forward(plan.line_joints[-1])[0] - np.array(end)), 5e-5)

    def test_upright_start_gets_small_joint_move_then_long_straight_line(self):
        plan = plan_line(UPRIGHT, [0.30, -0.10, 0.40])
        self.assertTrue(plan.needs_joint_move)
        # A line the controller can drive (constant arm angle, joints within speed)
        # starts farther from the upright singularity than an unconstrained one would.
        self.assertLess(plan.joint_move_deg, 60)
        self.assertGreater(plan.straight_mm, 0.7 * plan.total_mm)
        self.assertLessEqual(plan.peak_joint_speed_deg_s, self.settings.joint_speed_limit + 1e-9)
        # The line starts on the segment between the measured start and the target.
        self.assertLess(segment_distance_mm(plan.entry_pose[:3], plan.start_pose[:3], plan.target_pose[:3]), 0.1)
        self.assert_line_is_straight_and_usable(plan)

    def test_line_keeps_one_orientation_unless_blending_is_requested(self):
        # Pointing the tool down from upright forces a large change of orientation.
        target = [-0.35, 0.0, 0.30]
        fixed = plan_line(UPRIGHT, target, aim=DOWN)
        self.assertGreater(fixed.orientation_change_deg, 30)  # Done in the joint move.
        self.assert_line_is_straight_and_usable(fixed)
        blended = plan_line(UPRIGHT, target, aim=DOWN, settings=PlanSettings(rotate_on_line=True))
        self.assertGreater(rotation_angle(forward(blended.line_joints[0])[1],
                                          forward(blended.line_joints[-1])[1]), math.radians(30))
        self.assert_line_is_straight_and_usable(blended, fixed_orientation=False)

    def test_free_orientation_from_usable_start_is_a_pure_translation(self):
        position, _ = forward(BENT)
        plan = plan_line(BENT, position + [0.10, 0.15, -0.10])
        self.assertFalse(plan.needs_joint_move)
        self.assertLess(plan.orientation_change_deg, 0.05)
        self.assert_line_is_straight_and_usable(plan)

    def test_usable_start_needs_no_joint_move(self):
        position, rotation = forward(BENT)
        plan = plan_line(BENT, position + [0.10, 0.15, -0.10], keep_orientation=True)
        self.assertFalse(plan.needs_joint_move)
        self.assertEqual(plan.line_joints[0], BENT)
        self.assertAlmostEqual(plan.straight_mm, plan.total_mm, places=6)
        self.assertLess(plan.orientation_change_deg, 0.05)
        self.assertLess(rotation_angle(rotation, rpy_to_matrix(*plan.target_pose[3:])), 1e-3)
        self.assert_line_is_straight_and_usable(plan)

    def test_orientation_choices_from_upright(self):
        target = [-0.45, 0.0, 0.15]
        down = plan_line(UPRIGHT, target, aim=DOWN)
        tool = rpy_to_matrix(*down.target_pose[3:]) @ np.array(DOWN[0], dtype=float)
        self.assertGreater(float(tool @ np.array(DOWN[1])), math.cos(math.radians(0.1)))
        self.assert_line_is_straight_and_usable(down)
        wanted = rpy_to_matrix(*down.target_pose[3:])
        fixed = plan_line(UPRIGHT, target, rotation=wanted)
        self.assertLess(rotation_angle(wanted, rpy_to_matrix(*fixed.target_pose[3:])), 1e-3)

    def test_same_request_gives_same_plan(self):
        first, second = (plan_line(UPRIGHT, [-0.30, 0.20, 0.25]) for _ in range(2))
        self.assertEqual(first, second)

    def test_joint_only_plan_has_no_line(self):
        plan = plan_joint_move(UPRIGHT, [-0.35, 0.0, 0.30])
        self.assertTrue(plan.needs_joint_move)
        self.assertEqual(len(plan.line_joints), 1)
        self.assertEqual(plan.line_joints[0], plan.entry_joints)
        self.assertLess(np.linalg.norm(forward(plan.entry_joints)[0] - [-0.35, 0.0, 0.30]), 5e-5)

    def test_start_below_the_height_limit_may_move_up(self):
        # Flange 34 mm above the base plane (2026-11-23). Moving away upward is allowed.
        low = np.radians([37.616, 57.185, -1.369, 122.024, -8.082, 3.174, -4.643])
        self.assertLess(height_floor_mm(low, 50), 50)
        plan = plan_line(low, [-0.30, 0.0, 0.35])
        self.assertGreaterEqual(plan.min_z_mm, height_floor_mm(low, 50))
        self.assertEqual(height_floor_mm(UPRIGHT, 50), 50)

    def test_unreachable_low_or_conflicting_requests_are_rejected(self):
        with self.assertRaises(PlanError):
            plan_line(UPRIGHT, [0.9, 0.0, 0.3])
        with self.assertRaisesRegex(PlanError, "최저 높이"):
            plan_line(UPRIGHT, [-0.30, 0.0, 0.02])
        with self.assertRaises(PlanError):
            plan_line([3.0] + [0.0] * 6, [-0.35, 0.0, 0.30])  # J1 outside its limit.
        with self.assertRaises(ValueError):
            plan_line(UPRIGHT, [-0.35, 0.0, 0.30], keep_orientation=True, aim=DOWN)
        for bad in ([float("nan"), 0, 0.3], [0, 0.3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_line(UPRIGHT, bad)
        with self.assertRaises(ValueError):
            plan_line(UPRIGHT, [-0.35, 0.0, 0.30], settings=PlanSettings(step_mm=0))


if __name__ == "__main__":
    unittest.main()
