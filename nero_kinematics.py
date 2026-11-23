"""Offline Nero kinematics: FK, Jacobian, numerical IK and straight-line planning.

This module never opens CAN and never talks to a robot. It uses the SDK's
modified-DH table, so every result is a *model* prediction. The controller's
own IK may pick another joint solution for the same flange pose, and nothing
here checks collisions with the table, the tool, or the arm itself.
"""

from dataclasses import dataclass
import math

import numpy as np

from pyAgxArm import ArmModel, create_agx_arm_config
from pyAgxArm.utiles.mdh_kinematics import get_mdh


MDH = get_mdh("nero")
TOOL_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
             "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}


class PlanError(ValueError):
    """The requested motion cannot be planned with this model and these limits."""


def default_limits():
    limits = create_agx_arm_config(robot=ArmModel.NERO)["joint_limits"]
    return limits_array(limits)


def limits_array(limits):
    """Accept the SDK's {"joint1": [low, high], ...} mapping or a 7x2 array."""
    if isinstance(limits, dict):
        limits = [limits["joint{}".format(i)] for i in range(1, 8)]
    limits = np.array(limits, dtype=float)
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("관절 제한은 [하한, 상한] 7쌍이어야 합니다.")
    return limits


def _joints(q):
    q = np.array(q, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("관절각은 유한한 rad 값 7개여야 합니다.")
    return q


def link_frames(q):
    """Base-to-link transforms, same convention as pyAgxArm's fk_from_mdh."""
    frames, total = [], np.eye(4)
    for (d, a, alpha, offset), angle in zip(MDH, _joints(q)):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(angle + offset), math.sin(angle + offset)
        total = total @ np.array([
            [ct, -st, 0.0, a],
            [ca * st, ca * ct, -sa, -sa * d],
            [sa * st, sa * ct, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0]])
        frames.append(total)
    return frames


def forward(q):
    """Flange position (m) and rotation matrix in the base frame."""
    tip = link_frames(q)[-1]
    return tip[:3, 3].copy(), tip[:3, :3].copy()


def jacobian(q):
    """6x7 geometric Jacobian: rows 0-2 linear (m/rad), rows 3-5 angular."""
    return _jacobian_from(link_frames(q))


def _jacobian_from(frames):
    tip = frames[-1][:3, 3]
    result = np.zeros((6, 7))
    for i, frame in enumerate(frames):
        axis = frame[:3, 2]  # Modified DH: joint i turns about its own z axis.
        result[:3, i] = _cross(axis, tip - frame[:3, 3])
        result[3:, i] = axis
    return result


def min_singular_value(q):
    """Smallest singular value of the 6x7 Jacobian; near 0 means singular."""
    return float(np.linalg.svd(jacobian(q), compute_uv=False)[-1])


def rpy_to_matrix(roll, pitch, yaw):
    """SDK pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(rotation):
    """Inverse of rpy_to_matrix; mirrors the SDK's handling of pitch = +/-90 deg."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    if abs(math.cos(pitch)) < 1e-9:
        return [0.0, pitch, math.atan2(-rotation[0, 1], rotation[1, 1])]
    return [math.atan2(rotation[2, 1], rotation[2, 2]), pitch,
            math.atan2(rotation[1, 0], rotation[0, 0])]


def pose6(position, rotation):
    """[x, y, z, roll, pitch, yaw] in m/rad, the format move_l/move_p expect."""
    return [float(v) for v in position] + matrix_to_rpy(rotation)


def rotation_vector(rotation):
    """Axis * angle of a rotation matrix (the SO(3) logarithm)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-9:
        return skew / 2.0
    if math.pi - angle < 1e-6:
        # sin(angle) vanishes; recover the axis from the symmetric part.
        symmetric = (rotation + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(symmetric)))
        axis = symmetric[:, k] / math.sqrt(symmetric[k, k])
        return (-axis if float(axis @ skew) < 0 else axis) * angle
    return skew * angle / (2.0 * math.sin(angle))


def from_rotation_vector(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = np.asarray(vector) / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def rotation_angle(a, b):
    """Angle (rad) between two rotation matrices."""
    return float(np.linalg.norm(rotation_vector(a.T @ b)))


def segment_distance_mm(point, a, b):
    """Distance in mm from a point to the segment a-b (all in m)."""
    point, a, b = (np.asarray(v, dtype=float) for v in (point, a, b))
    span = b - a
    length2 = float(span @ span)
    t = 0.0 if length2 < 1e-18 else min(1.0, max(0.0, float((point - a) @ span) / length2))
    return float(np.linalg.norm(point - (a + t * span))) * 1000.0


# Nero is an S-R-S arm (all MDH link offsets a = 0): J1-J3 axes meet at the
# shoulder, J4 is the elbow, J5-J7 axes meet at the wrist. Indices into link_frames().
SHOULDER, ELBOW, WRIST = 1, 2, 4
UP = np.array([0.0, 0.0, 1.0])


def _cross(a, b):
    """3-vector cross product; np.cross is many times slower for single vectors."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _swivel(shoulder, elbow, wrist):
    """Angle of the elbow about the shoulder-wrist axis, from the vertical plane through it."""
    axis = (wrist - shoulder) / float(np.linalg.norm(wrist - shoulder))
    offset = (elbow - shoulder) - float((elbow - shoulder) @ axis) * axis
    reference = UP - float(axis[2]) * axis
    reference /= max(1e-12, float(np.linalg.norm(reference)))
    return math.atan2(float(_cross(reference, offset) @ axis), float(reference @ offset))


def arm_angle(q):
    """Elbow swivel (rad) about the shoulder-wrist axis.

    Firmware 1.12 holds this angle constant during move_l: in the two
    successful lines of 2026-11-16 it stayed within 0.5 degrees, and the joints
    matched a constant-angle simulation within about 0.3 degrees (median).
    The reference plane is vertical, so the angle is ill-defined when the
    shoulder-wrist axis is vertical (see wrist_tilt).
    """
    return _swivel_from(link_frames(q))


def _swivel_from(frames):
    return _swivel(frames[SHOULDER][:3, 3], frames[ELBOW][:3, 3], frames[WRIST][:3, 3])


def wrist_tilt(q):
    """Angle (rad) between the shoulder-wrist axis and the vertical."""
    frames = link_frames(q)
    axis = frames[WRIST][:3, 3] - frames[SHOULDER][:3, 3]
    return math.atan2(math.hypot(float(axis[0]), float(axis[1])), float(axis[2]))


def _point_jacobian(frames, index):
    """3x7 position Jacobian of the origin of link_frames()[index]."""
    point = frames[index][:3, 3]
    result = np.zeros((3, 7))
    for i in range(index + 1):  # Joint i+1 turns about frames[i]'s z axis.
        result[:, i] = _cross(frames[i][:3, 2], point - frames[i][:3, 3])
    return result


def arm_angle_gradient(q):
    """d(arm_angle)/dq as a length-7 vector."""
    return _swivel_gradient_from(link_frames(q))


def _swivel_gradient_from(frames):
    s, e, w = frames[SHOULDER][:3, 3], frames[ELBOW][:3, 3], frames[WRIST][:3, 3]
    h = 1e-7
    partial = np.zeros(6)  # d(angle)/d(elbow xyz, wrist xyz); the shoulder never moves.
    for k in range(6):
        plus, minus = [e.copy(), w.copy()], [e.copy(), w.copy()]
        plus[k // 3][k % 3] += h
        minus[k // 3][k % 3] -= h
        partial[k] = _wrap(_swivel(s, *plus) - _swivel(s, *minus)) / (2 * h)
    return partial[:3] @ _point_jacobian(frames, ELBOW) + partial[3:] @ _point_jacobian(frames, WRIST)


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _perpendicular(axis):
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    result = np.cross(axis, helper)
    return result / float(np.linalg.norm(result))


def _aim_error(rotation, tool_axis, direction):
    """Rotation vector turning the flange's tool axis onto a base direction."""
    current = rotation @ tool_axis
    cross = np.cross(current, direction)
    sine, cosine = float(np.linalg.norm(cross)), float(current @ direction)
    if sine < 1e-9:
        if cosine > 0:
            return np.zeros(3), current
        cross, sine = _perpendicular(current), 1.0  # Exactly opposite: any turn axis works.
    return cross / sine * math.atan2(sine, cosine), current


def solve_ik(position, seed, limits, *, rotation=None, aim=None, reference=None, swivel=None,
             iterations=200, position_tolerance=2e-5, angle_tolerance=2e-4):
    """Damped least-squares IK with a null-space pull toward ``reference``.

    ``rotation`` (3x3) constrains the full orientation. ``aim`` =
    (flange-frame axis, base-frame direction) constrains one axis only, leaving
    the spin about it free. With neither, only the position is constrained.
    ``swivel`` (rad) also fixes the arm angle, as the controller does in move_l.
    Returns (joints, converged). Joints are always clipped into ``limits``.
    """
    position = np.asarray(position, dtype=float)
    q = np.clip(_joints(seed), limits[:, 0], limits[:, 1])
    reference = q.copy() if reference is None else _joints(reference)
    for _ in range(iterations):
        frames = link_frames(q)  # Computed once per iteration and shared below.
        current, orientation = frames[-1][:3, 3], frames[-1][:3, :3]
        full = _jacobian_from(frames)
        error, task = position - current, full[:3]
        angle_error = 0.0
        if rotation is not None:
            twist = rotation_vector(rotation @ orientation.T)
            angle_error = float(np.linalg.norm(twist))
            error, task = np.concatenate([error, twist]), full
        elif aim is not None:
            twist, axis = _aim_error(orientation, aim[0], aim[1])
            angle_error = float(np.linalg.norm(twist))
            # Spin about the axis is free: keep only the two directions across it,
            # so the task stays full rank (a 3-row projector would always be singular).
            first = _perpendicular(axis)
            across = np.vstack([first, np.cross(axis, first)])
            error, task = np.concatenate([error, across @ twist]), np.vstack([full[:3], across @ full[3:]])
        if swivel is not None:
            swivel_error = _wrap(swivel - _swivel_from(frames))
            angle_error = max(angle_error, abs(swivel_error))
            error, task = np.append(error, swivel_error), np.vstack([task, _swivel_gradient_from(frames)])
        if float(np.linalg.norm(error[:3])) < position_tolerance and angle_error < angle_tolerance:
            return q, True
        # Bound each linearized step: at most 5 cm and about 17 degrees.
        scale = min(1.0, 0.05 / max(1e-12, float(np.linalg.norm(error[:3]))))
        if angle_error > 0.3:
            scale = min(scale, 0.3 / angle_error)
        smallest = float(np.linalg.svd(task, compute_uv=False)[-1])
        damping = 1e-8 if smallest >= 0.03 else (1.0 - (smallest / 0.03) ** 2) * 0.05 ** 2 + 1e-8
        inverse = task.T @ np.linalg.inv(task @ task.T + damping * np.eye(task.shape[0]))
        step = inverse @ (error * scale) + (np.eye(7) - inverse @ task) @ (0.5 * (reference - q))
        largest = float(np.max(np.abs(step)))
        if largest > 0.2:
            step *= 0.2 / largest
        q = np.clip(q + step, limits[:, 0], limits[:, 1])
    return q, False


@dataclass
class PlanSettings:
    step_mm: float = 5.0             # Spacing of the checked points on the line.
    step_deg: float = 2.0            # Same, for orientation change along the line.
    max_joint_step_deg: float = 4.0  # Largest joint change allowed between two points.
    min_sigma: float = 0.05          # Singularity guard on the 6x7 Jacobian.
    limit_margin_deg: float = 2.0    # Stay this far inside the SDK joint limits.
    min_z_mm: float = 50.0           # Lowest allowed height of flange/elbow/wrist origins.
    seeds: int = 24                  # IK restarts used to find target postures.
    # The straight part keeps the shoulder-wrist axis at least this far from vertical,
    # where the controller's arm-angle reference is undefined. Lines starting at 1.9
    # degrees failed (23:19); of five starting at 4.3-5.4 degrees, three let the arm
    # angle slip 1-26 degrees in the first 20 mm (tilt about 6 there) and drifted up
    # to 10.8 mm (00:06). The line that started at 47 degrees held it to 0.5.
    min_tilt_deg: float = 8.0
    # False: the straight part is a pure translation and any reorientation
    # happens in the joint move. True blends the orientation along the line;
    # firmware 1.12 crawled and then reported NO_SOLUTION on such a line (2026-11-16).
    rotate_on_line: bool = False
    # True: the joint move is led along the straight joint-space line so all
    # joints arrive together. False: one move_j (joints finish at different times).
    joints_together: bool = True
    # move_l speed model, per percent of speed. Measured at 5% on 2026-11-16/17: the
    # flange cruises at about 45 mm/s after ramping at about 40 mm/s^2, and the
    # controller barely slows the flange when a joint cannot keep up. In the 00:07
    # run J1/J5 stayed near 9 deg/s while the line needed 10-15 deg/s for 2 s,
    # and the flange drifted 1.4 mm (at 10 deg/s needed) to 16 mm (at 14). Brief
    # needs of 12-18 deg/s were met in other runs (J4 13.6, J5 18.6 deg/s), so the
    # limit is 12 deg/s at 5%: a few mm of drift is possible near it. The same move
    # at 7% and 5% (2026-11-17 17:51 / 18:20) scaled joint speed (12.9 vs 9.2 deg/s)
    # and peak flange speed (75 vs 53 mm/s) by 7:5, but the flange ramp stayed about
    # 38 mm/s^2 at both, so the ramp is a constant and not per percent.
    speed_percent: int = 5
    flange_speed_per_percent: float = 9.0     # mm/s
    flange_accel: float = 40.0                # mm/s^2, independent of the speed percentage
    joint_speed_per_percent: float = 2.4      # deg/s a joint is planned to need at most

    def validate(self):
        values = (self.step_mm, self.step_deg, self.max_joint_step_deg, self.min_sigma,
                  self.limit_margin_deg, self.min_z_mm, self.min_tilt_deg, self.flange_speed_per_percent,
                  self.flange_accel, self.joint_speed_per_percent)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("계획 설정은 유한한 숫자여야 합니다.")
        if not (1 <= self.step_mm <= 20 and 0.5 <= self.step_deg <= 10
                and 0.5 <= self.max_joint_step_deg <= 15 and 0.005 <= self.min_sigma <= 0.2
                and 0 <= self.limit_margin_deg <= 10 and 1 <= self.seeds <= 200
                and 0 <= self.min_tilt_deg <= 30 and 1 <= self.speed_percent <= 100
                and min(self.flange_speed_per_percent, self.flange_accel,
                        self.joint_speed_per_percent) > 0):
            raise ValueError("계획 설정이 허용 범위를 벗어났습니다.")

    def flange_speed(self, from_start_mm, to_end_mm):
        """Expected move_l flange speed (mm/s) at a point of a line, ramps included."""
        top = self.flange_speed_per_percent * self.speed_percent
        accel = self.flange_accel
        return min(top, math.sqrt(2 * accel * max(0.0, from_start_mm)),
                   math.sqrt(2 * accel * max(0.0, to_end_mm)))

    @property
    def joint_speed_limit(self):
        return self.joint_speed_per_percent * self.speed_percent


@dataclass
class Plan:
    """A joint move to ``entry`` (if needed), then a straight flange line to the target."""
    start_joints: list         # rad, as measured
    start_pose: list           # m/rad, FK of start_joints
    target_pose: list          # m/rad, what move_l will be asked to reach
    entry_joints: list         # rad, where the straight part begins
    entry_pose: list           # m/rad, FK of entry_joints
    line_joints: list          # predicted joints along the straight part, entry first
    needs_joint_move: bool
    total_mm: float            # start to target, straight distance
    straight_mm: float         # entry to target: the part driven with move_l
    joint_move_deg: float      # largest single-joint change in the joint move
    joint_move_path_mm: float  # flange path length of the joint move (curved)
    min_sigma: float           # worst singularity margin on the straight part
    max_joint_step_deg: float  # largest joint change between checked line points
    min_z_mm: float            # lowest flange/elbow/wrist origin over the whole motion
    orientation_change_deg: float
    arm_angle_deg: float = 0.0  # elbow swivel the controller holds during the straight part
    min_tilt_deg: float = 0.0   # smallest shoulder-wrist tilt from vertical on the straight part
    peak_joint_speed_deg_s: float = 0.0  # fastest joint speed the straight part needs (speed model)


def _lowest_origin_mm(q):
    # Frames 1-2 sit at the fixed shoulder; 3-4 are the elbow, 5-6 the wrist, 7 the flange.
    return min(float(frame[2, 3]) for frame in link_frames(q)[2:]) * 1000.0


def _problem(q, limits, settings):
    """Why a posture is unusable for the straight part, or None."""
    margin = math.radians(settings.limit_margin_deg)
    for i, (value, (low, high)) in enumerate(zip(q, limits), 1):
        if not low + margin <= value <= high - margin:
            return "J{} 관절 제한 여유 부족".format(i)
    if min_singular_value(q) < settings.min_sigma:
        return "특이점 근접"
    if _lowest_origin_mm(q) < settings.min_z_mm:
        return "최저 높이 한도 미만"
    return None


def _line_samples(p0, r0, p1, r1, settings):
    twist = rotation_vector(r0.T @ r1)
    count = max(1, math.ceil(float(np.linalg.norm(p1 - p0)) * 1000.0 / settings.step_mm),
                math.ceil(math.degrees(float(np.linalg.norm(twist))) / settings.step_deg))
    return [(p0 + (p1 - p0) * i / count, r0 @ from_rotation_vector(twist * i / count))
            for i in range(count + 1)]


def _line_problem(q, limits, settings):
    """_problem plus what only matters while the controller drives move_l."""
    problem = _problem(q, limits, settings)
    if problem is None and math.degrees(wrist_tilt(q)) < settings.min_tilt_deg:
        problem = "어깨-손목 축이 수직에 가까움"
    return problem


def _walk(samples, order, q, limits, settings):
    """Follow consecutive line points from posture ``q``; stop at the first bad one.

    Each point keeps the arm angle of ``q``, because that is the joint path the
    controller itself takes during move_l (see arm_angle).
    """
    swivel = arm_angle(q)
    path = []
    for index in order:
        position, rotation = samples[index]
        candidate, converged = solve_ik(position, q, limits, rotation=rotation, swivel=swivel,
                                        iterations=60)
        if (not converged
                or float(np.max(np.abs(candidate - q))) > math.radians(settings.max_joint_step_deg)
                or _line_problem(candidate, limits, settings) is not None):
            break
        path.append(candidate)
        q = candidate
    return path


def _segments(line):
    """Flange travel (mm) and largest joint change (deg) between consecutive postures."""
    points = np.array([forward(q)[0] for q in line]) * 1000.0
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    jumps = np.degrees(np.max(np.abs(np.diff(np.array(line), axis=0)), axis=1))
    return steps, jumps


def _peak_joint_speed(steps, jumps, settings):
    """Fastest joint speed (deg/s) move_l needs along these segments, under the speed model."""
    length, travelled, worst = float(np.sum(steps)), 0.0, 0.0
    for step, jump in zip(steps, jumps):
        middle = travelled + step / 2.0
        travelled += step
        if step > 1e-9:
            worst = max(worst, jump / step * settings.flange_speed(middle, length - middle))
    return worst


def _longest_fast_enough(line, settings):
    """The longest tail of ``line`` (ending at its last posture) the joints can follow."""
    steps, jumps = _segments(line)
    for first in range(len(line) - 1):
        if _peak_joint_speed(steps[first:], jumps[first:], settings) <= settings.joint_speed_limit:
            return line[first:]
    return line[-1:]


def _joint_move_trace(start, end, together):
    """Flange path length (mm) and lowest origin (mm) of a joint move from start to end.

    ``together``: the joints are led along the straight joint-space line and
    arrive together. Otherwise it is one move_j, which firmware 1.12 does not
    time-synchronize: in the 2026-11-16 run all joints turned at about the same
    rate (9 deg/s at 5%) and each stopped when it arrived. The length follows
    the chosen kind of path; the lowest point covers both.
    """
    travel = end - start
    largest = float(np.max(np.abs(travel)))
    lowest, length, previous = _lowest_origin_mm(start), 0.0, forward(start)[0]
    for i in range(1, 51):
        on_line = start + travel * i / 50
        equal_rate = start + np.sign(travel) * np.minimum(np.abs(travel), largest * i / 50)
        point = forward(on_line if together else equal_rate)[0]
        length += float(np.linalg.norm(point - previous)) * 1000.0
        previous = point
        lowest = min(lowest, _lowest_origin_mm(equal_rate), _lowest_origin_mm(on_line))
    return length, lowest


def joint_move_trace(start_joints, end_joints, together=True):
    """(flange path length mm, lowest elbow/wrist/flange origin mm) of a joint move.

    For moves that do not go through plan_line, such as nero_home.py. The
    lowest point includes the start posture; see height_floor_mm.
    """
    return _joint_move_trace(_joints(start_joints), _joints(end_joints), together)


def height_floor_mm(start_joints, min_z_mm):
    """Lowest height (mm) a move from ``start_joints`` may reach.

    ``min_z_mm``, unless the arm already starts lower: then it may not go below
    where it is now (1 mm of slack). On 2026-11-23 a return home that only rose
    from a flange 34 mm above the base plane was refused, because the start
    itself counted as going below the 50 mm limit.
    """
    return min(min_z_mm, _lowest_origin_mm(_joints(start_joints)) - 1.0)


def height_limit_text(floor_mm, min_z_mm):
    """Korean description of the limit height_floor_mm returned, for messages."""
    if floor_mm < min_z_mm:
        return "한도 {:g}mm보다 낮은 곳에서 출발 → 출발 높이 {:.0f}mm보다 내려가지 않으면 허용".format(
            min_z_mm, floor_mm + 1.0)
    return "한도 {:g}mm".format(min_z_mm)


def _prepare(start_joints, target_xyz, limits, rotation, aim, keep_orientation, settings):
    """Validate a request and find the usable joint postures at the target."""
    settings = settings or PlanSettings()
    settings.validate()
    limits = default_limits() if limits is None else limits_array(limits)
    start = _joints(start_joints)
    target = np.array(target_xyz, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("목표 좌표는 유한한 X, Y, Z 세 값입니다.")
    if sum(option is not None for option in (rotation, aim)) + bool(keep_orientation) > 1:
        raise ValueError("목표 방향 지정 방식은 하나만 선택하세요.")
    if np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
        raise PlanError("현재 관절각이 SDK 제한 범위 밖입니다.")
    if keep_orientation:
        rotation = forward(start)[1]
    if rotation is not None:
        rotation = np.array(rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("목표 방향은 3x3 회전 행렬이어야 합니다.")
    if aim is not None:
        aim = tuple(np.array(v, dtype=float) / float(np.linalg.norm(v)) for v in aim)

    generator = np.random.default_rng(0)  # Fixed seed: the same request plans the same way.
    seeds = [start.copy()]
    for elbow in (60.0, 90.0, 30.0):
        for shoulder in (30.0, -30.0, 60.0, -60.0):
            bent = start.copy()
            bent[1], bent[3] = math.radians(shoulder), math.radians(elbow)
            seeds.append(bent)
    while len(seeds) < settings.seeds:
        seeds.append(generator.uniform(limits[:, 0], limits[:, 1]))
    # With a free orientation, postures that keep the current one come first:
    # they allow a straight line with no reorientation at all.
    requests = [dict(rotation=rotation, aim=aim)]
    if rotation is None and aim is None:
        requests.insert(0, dict(rotation=forward(start)[1]))
    postures, reasons = [], set()
    for request in requests:
        for seed in seeds[:settings.seeds]:
            q, converged = solve_ik(target, seed, limits, reference=seed, **request)
            reason = _problem(q, limits, settings) if converged else "IK 해 없음"
            if reason is not None:
                reasons.add(reason)
            elif all(float(np.max(np.abs(q - other))) > math.radians(3) for other in postures):
                postures.append(q)
    if not postures:
        raise PlanError("목표 지점에서 사용할 수 있는 관절 자세를 찾지 못했습니다 ({}). "
                        "좌표, 방향 지정, 최저 높이 한도를 확인하세요.".format(", ".join(sorted(reasons))))
    return settings, limits, start, target, postures


def _finish(start, target, line, trace, settings):
    p0, r0 = forward(start)
    entry = line[0]
    entry_position, entry_rotation = forward(entry)
    end_rotation = forward(line[-1])[1]
    steps = [float(np.max(np.abs(b - a))) for a, b in zip(line, line[1:])]
    return Plan(
        start_joints=start.tolist(), start_pose=pose6(p0, r0),
        target_pose=pose6(target, end_rotation),
        entry_joints=entry.tolist(), entry_pose=pose6(entry_position, entry_rotation),
        line_joints=[q.tolist() for q in line],
        needs_joint_move=float(np.max(np.abs(entry - start))) > math.radians(0.2),
        total_mm=float(np.linalg.norm(target - p0)) * 1000.0,
        straight_mm=float(np.linalg.norm(target - entry_position)) * 1000.0,
        joint_move_deg=math.degrees(float(np.max(np.abs(entry - start)))),
        joint_move_path_mm=trace[0],
        min_sigma=min(min_singular_value(q) for q in line),
        max_joint_step_deg=math.degrees(max(steps)) if steps else 0.0,
        min_z_mm=min([trace[1]] + [_lowest_origin_mm(q) for q in line]),
        orientation_change_deg=math.degrees(rotation_angle(r0, end_rotation)),
        arm_angle_deg=math.degrees(arm_angle(entry)),
        min_tilt_deg=min(math.degrees(wrist_tilt(q)) for q in line),
        peak_joint_speed_deg_s=_peak_joint_speed(*_segments(line), settings) if len(line) > 1 else 0.0)


def plan_line(start_joints, target_xyz, limits=None, *, rotation=None, aim=None,
              keep_orientation=False, settings=None):
    """Plan a straight flange line from the current posture to ``target_xyz`` (m).

    Orientation at the target: ``rotation`` (3x3) fixes it, ``keep_orientation``
    reuses the starting one, ``aim`` = (flange axis, base direction) fixes one
    axis, and with none of these the IK is free to choose. If the line cannot
    start at the current posture (typically a singular, fully stretched arm),
    the plan begins with one joint move to a usable point on the line.
    """
    settings, limits, start, target, postures = _prepare(
        start_joints, target_xyz, limits, rotation, aim, keep_orientation, settings)
    p0, r0 = forward(start)
    start_usable = _line_problem(start, limits, settings) is None
    best, tried, ends_vertical = None, [], 0
    for posture in postures:
        if _line_problem(posture, limits, settings) is not None:
            ends_vertical += math.degrees(wrist_tilt(posture)) < settings.min_tilt_deg
            continue  # The line has to end here, so it must suit move_l too.
        end_rotation = forward(posture)[1]
        turn = rotation_angle(r0, end_rotation)
        samples = _line_samples(p0, r0 if settings.rotate_on_line else end_rotation,
                                target, end_rotation, settings)
        if (start_usable and (settings.rotate_on_line or turn < math.radians(0.5))
                and all(rotation_angle(end_rotation, other) >= math.radians(1) for other in tried)):
            # Preferred: the whole line, starting right at the measured posture.
            tried.append(end_rotation)
            path = _walk(samples, range(1, len(samples)), start, limits, settings)
            if (len(path) == len(samples) - 1
                    and _peak_joint_speed(*_segments([start] + path), settings) <= settings.joint_speed_limit):
                return _finish(start, target, [start] + path, (0.0, _lowest_origin_mm(start)), settings)
        # Otherwise follow the line backward from the target as far as it stays usable
        # and the joints can keep up with the controller's flange speed.
        path = _walk(samples, range(len(samples) - 2, -1, -1), posture, limits, settings)
        line = _longest_fast_enough(list(reversed(path)) + [posture], settings)
        trace = _joint_move_trace(start, line[0], settings.joints_together)
        if trace[1] < height_floor_mm(start, settings.min_z_mm):
            continue
        # Trade lost straight distance against the size of the joint move that
        # reaches the line: 1 degree of the largest joint change counts as 2 mm.
        cost = (float(np.linalg.norm(forward(line[0])[0] - p0)) * 1000.0
                + 2.0 * math.degrees(float(np.max(np.abs(line[0] - start)))))
        if best is None or cost < best[0]:
            best = (cost, line, trace)
    if best is None and ends_vertical == len(postures):
        raise PlanError("목표 자세에서 어깨-손목 축이 수직에서 {:g}도 안쪽입니다. 컨트롤러가 직선 이동을 "
                        "끝낼 수 없는 자세이므로 --path joint(관절 이동, 곡선 경로)를 사용하세요.".format(
                            settings.min_tilt_deg))
    if best is None:
        raise PlanError("직선을 끝낼 수 있는 목표 자세가 없거나, 직선 시작점까지의 관절 이동이 "
                        "최저 높이 한도 {:g}mm 아래로 내려갑니다.".format(settings.min_z_mm))
    return _finish(start, target, best[1], best[2], settings)


def plan_joint_move(start_joints, target_xyz, limits=None, *, rotation=None, aim=None,
                    keep_orientation=False, settings=None):
    """Plan one joint-space move to ``target_xyz``: smooth, but the flange path curves."""
    settings, limits, start, target, postures = _prepare(
        start_joints, target_xyz, limits, rotation, aim, keep_orientation, settings)
    for posture in sorted(postures, key=lambda q: float(np.max(np.abs(q - start)))):
        trace = _joint_move_trace(start, posture, settings.joints_together)
        if trace[1] >= height_floor_mm(start, settings.min_z_mm):
            return _finish(start, target, [posture], trace, settings)
    raise PlanError("관절 이동 경로가 최저 높이 한도 {:g}mm 아래로 내려갑니다.".format(settings.min_z_mm))
