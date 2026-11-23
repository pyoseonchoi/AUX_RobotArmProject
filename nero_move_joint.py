"""Beginner step 3: move one already-enabled joint by 0.5 to 5 degrees."""

import argparse
import bisect
import math
import time

from nero_enable import DRIVER_ERROR_FIELDS, recent
from nero_diagnostics import diagnose
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, print_status, require_feedback


# Planning refusals the controller reports about a commanded path:
# NO_SOLUTION, SINGULARITY_POINT, TARGET_POS_EXCEEDS_LIMIT. They are not
# faults of the arm, and firmware 1.12 keeps the flag through e-stop, reset,
# disable and enable (2026-11-16 logs), so a move may have to start with it set.
REJECTIONS = (2, 3, 4)


class ControllerRejected(RuntimeError):
    """The controller itself refused or abandoned the commanded path."""


def _motion_snapshot(robot, *, newer_than=None):
    joint = robot.get_joint_angles()
    status = robot.get_arm_status()
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    required = [("관절 통합값", joint), ("로봇 상태", status)]
    required.extend((f"J{i} 드라이버", driver) for i, driver in enumerate(drivers, 1))
    # This SDK assembles seven angles from four packets. A new J7 packet
    # alone must not make an absent/stale J1-J6 packet appear current.
    parser = getattr(robot, "_parser", None)
    if parser is not None:
        required.extend((name, getattr(parser, attr, None)) for name, attr in (
            ("J1/J2 각도", "joint_12"), ("J3/J4 각도", "joint_34"),
            ("J5/J6 각도", "joint_56"), ("J7 각도", "joint_7"),
        ))
    require_feedback(required, newer_than=newer_than)
    # A planning refusal is judged by the caller (see RejectionWatch): only it
    # knows whether the flag belongs to the current target or an older one.
    if status.msg.ctrl_mode != 1 or int(status.msg.arm_status) not in (0,) + REJECTIONS:
        if status.msg.arm_status == 1:
            raise RuntimeError("EMERGENCY_STOP 상태입니다. nero_reset.py로 먼저 조회하세요. 이동/enable로 해제하지 않습니다.")
        if status.msg.ctrl_mode == 6:
            raise RuntimeError("교시 제어 모드가 남아 있습니다. 팔을 지지하고 python nero_teach.py control --firmware v112 --execute로 복귀하세요.")
        raise RuntimeError(
            f"정상 CAN 제어 상태가 아닙니다. ctrl_mode={status.msg.ctrl_mode}, arm_status={status.msg.arm_status}"
        )
    if status.msg.motion_status not in (0, 1):
        raise RuntimeError(f"알 수 없는 목표 도달 상태입니다: {status.msg.motion_status}")
    for i, driver in enumerate(drivers, 1):
        if not driver.msg.foc_status.driver_enable_status:
            raise RuntimeError(f"J{i} 활성화 피드백이 없습니다. 먼저 enable 단계를 실행하세요.")
        if any(getattr(driver.msg.foc_status, name) for name in DRIVER_ERROR_FIELDS):
            raise RuntimeError(f"J{i} 드라이버 오류가 보고됐습니다.")
    angles = list(joint.msg)
    if len(angles) != 7 or not all(math.isfinite(a) for a in angles):
        raise RuntimeError("7개 관절 각도를 확인하지 못했습니다.")
    return angles, status.msg.motion_status, joint.timestamp


def motion_feedback(robot):
    angles, motion_status, _ = _motion_snapshot(robot)
    return angles, motion_status


class RejectionWatch:
    """Tell a refusal of the target just sent from a flag left by an older command."""

    STALL_S = 2.0  # A leftover flag plus no motion for this long: the new target was not taken.

    def __init__(self, robot, start_angles):
        self.robot, self.start = robot, list(start_angles)
        self.leftover = int(robot.get_arm_status().msg.arm_status) in REJECTIONS
        self.sent = time.monotonic()
        self.gave_up = False

    def target_sent(self):
        self.sent = time.monotonic()

    def check(self, angles):
        """Raise once the controller has refused or abandoned the current target."""
        status = self.robot.get_arm_status()
        if int(status.msg.arm_status) not in REJECTIONS:
            self.leftover = False  # Cleared: any refusal from here on is about this target.
            return
        if not self.leftover:
            self.gave_up = True
            raise ControllerRejected("컨트롤러가 이 목표의 경로를 거부했습니다: {}. "
                                     "목표나 방향을 바꿔 다시 시도하세요.".format(status.msg.arm_status))
        moved = max(abs(a - b) for a, b in zip(angles, self.start))
        if time.monotonic() - self.sent >= self.STALL_S and moved < math.radians(0.05):
            self.gave_up = True
            raise ControllerRejected("이전 경로 거부 표시({})가 새 목표를 보낸 뒤에도 남아 있고 "
                                     "팔이 움직이지 않습니다.".format(status.msg.arm_status))

    def controller_gave_up(self):
        """True if, by its own fresh report, the controller is no longer driving to this target."""
        if self.gave_up:
            return True
        status = self.robot.get_arm_status()
        return recent(status) and not self.leftover and int(status.msg.arm_status) in REJECTIONS


class ProgressTimeout:
    """Expires after ``timeout`` seconds without getting closer to the target.

    A slow but advancing move is not cut off (and e-stopped) just because it
    takes long; a move that stops advancing still fails after ``timeout``.
    ``steps`` is the smallest improvement of each error that counts as progress.
    """

    def __init__(self, timeout, steps):
        self.timeout, self.steps = timeout, tuple(steps)
        self.best, self.since = None, time.monotonic()

    def expired(self, errors):
        now = time.monotonic()
        if self.best is None:
            self.best, self.since = list(errors), now
        for i, (error, step) in enumerate(zip(errors, self.steps)):
            if error < self.best[i] - step:
                self.best[i], self.since = error, now
        return now - self.since >= self.timeout


class ResendGuard:
    """Says when to send the same target again because part of it was not applied.

    A move_j goes out as four CAN frames (J1/J2, J3/J4, J5/J6, J7) and a move_l
    as three. On 2026-11-18 01:28 a teaching waypoint moved J7 but left J2
    exactly at the previous waypoint while the controller reported "reached"
    (motion_status 0): the J1/J2 frame was most likely not applied, and the old
    code waited 30 s and failed. Re-sending the same target is harmless.
    Resend when the controller says it is done, the measured errors are above
    tolerance and have not improved for WAIT_S since the last send.
    """

    WAIT_S = 0.6
    MAX_RESENDS = 3

    def __init__(self, steps):
        self.steps = tuple(steps)
        self.best, self.improved_at, self.sent_at, self.resends = None, time.monotonic(), time.monotonic(), 0

    def sent(self):
        self.sent_at = time.monotonic()

    def due(self, controller_done, errors):
        now = time.monotonic()
        if self.best is None:
            self.best = list(errors)
        for i, (error, step) in enumerate(zip(errors, self.steps)):
            if error < self.best[i] - step:
                self.best[i], self.improved_at = error, now
        if (controller_done and self.resends < self.MAX_RESENDS
                and now - self.improved_at >= self.WAIT_S and now - self.sent_at >= self.WAIT_S):
            self.resends += 1
            self.improved_at = now
            return True
        return False


def _verified_still(robot, seconds=0.5):
    """True only if fresh feedback keeps every joint within 0.05 degrees for the whole window."""
    started = time.monotonic()
    lows = highs = last = None
    fresh = 0
    while True:
        angles, _, stamp = _motion_snapshot(robot)  # Raises on stale feedback or a motor fault.
        lows = angles if lows is None else [min(a, b) for a, b in zip(lows, angles)]
        highs = angles if highs is None else [max(a, b) for a, b in zip(highs, angles)]
        if max(h - l for l, h in zip(lows, highs)) > math.radians(0.05):
            return False
        if last is None or stamp > last:
            fresh, last = fresh + 1, stamp
        if time.monotonic() - started >= seconds:
            return fresh >= 3
        time.sleep(0.05)


def stop_failed_move(robot, exc, watch):
    """End a move that failed after its target was sent. Returns True if an e-stop was requested.

    The electronic emergency stop is the SDK's only stop command, and recovering
    from it means a reset that drops the arm. So it is skipped in exactly one
    case: the controller itself reports that it refused or abandoned this
    target, and fresh feedback then shows all joints still for 0.5 s. A timeout,
    lost feedback, Ctrl+C or any other failure while the controller may still be
    driving the arm requests the stop immediately, without that wait.
    """
    reason = str(exc) or "사용자 중단"
    print("이동 실패:", type(exc).__name__, reason)
    still = False
    try:
        still = watch is not None and watch.controller_gave_up() and _verified_still(robot)
    except BaseException as check_error:  # Includes a second Ctrl+C: stop right away.
        print("정지 상태를 확인하지 못했습니다:", str(check_error) or type(check_error).__name__)
    if still:
        diagnose(robot, "emergency_stop_skipped", error=reason)
        print("컨트롤러가 이미 이 이동을 중단했고 0.5초 동안 관절이 멈춰 있었습니다. 전자 급정지는 보내지 않습니다.")
        print("모터는 활성화 상태로 남습니다. reset 없이 다른 목표로 다시 시도할 수 있습니다.")
        return False
    try:
        robot.electronic_emergency_stop()
        diagnose(robot, "emergency_stop_requested", error=reason)
        print("전자 급정지를 요청했습니다. 실제 정지 여부를 확인하세요.")
    except Exception as stop_error:
        print("전자 급정지 전송 실패:", stop_error)
        print("장비의 물리 비상정지 절차를 따르세요.")
    return True


def require_planned_mode(robot):
    status = robot.get_arm_status()
    require_feedback([("이동 모드", status)])
    if int(status.msg.mode_feedback) not in (0, 1, 2, 3):
        raise RuntimeError(
            "일반 위치 이동 모드가 아닙니다: {}. 목표는 보내지 않습니다. "
            "교시 모드는 control로 복귀하고, CAN/MIT 상태는 팔을 지지한 뒤 "
            "disable → nero_teach.py prepare --execute → enable 순서로 준비하세요."
            .format(status.msg.mode_feedback))


def wait_for_motion_feedback(robot, *, newer_than=None, timeout=2.0):
    """Wait only before issuing a target; feedback age remains below 0.5 s."""
    deadline = time.monotonic() + timeout
    last_reason = "이동 준비 피드백 대기 중"
    while True:
        if not robot.is_connected():
            raise RuntimeError("CAN 연결이 닫혀 있어 이동을 준비할 수 없습니다.")
        error = robot.get_comm_error()
        if error is not None:
            raise RuntimeError(f"CAN 통신 오류로 이동을 중단합니다: {error}")
        try:
            return _motion_snapshot(robot, newer_than=newer_than)
        except FeedbackUnavailableError as exc:
            last_reason = str(exc)
        if time.monotonic() >= deadline:
            reader = getattr(robot.get_context(), "_read_th", None)
            reader_alive = reader is not None and reader.is_alive()
            raise FeedbackUnavailableError(
                f"{timeout:g}초 안에 이동 준비 피드백을 받지 못했습니다. {last_reason}. "
                f"CAN 수신 스레드 실행 여부={reader_alive}"
            )
        time.sleep(0.02)


def check_before_move(robot):
    """Observe fresh joint feedback, without treating an old target flag as motion.

    This software check cannot guarantee a physical stop or prevent collisions.
    """
    print("이동 전 검사: 필요한 전체 피드백을 최대 2초 기다립니다.")
    wait_for_motion_feedback(robot, newer_than=feedback_time())
    print("전체 피드백 수신 완료. 0.5초 동안 관절 변화를 관찰합니다.")
    limits = robot.get_config()["joint_limits"]
    started = time.monotonic()
    lows = highs = None
    last_timestamp = None
    new_samples = 0
    tolerance = math.radians(0.05)
    while True:
        angles, motion_status, timestamp = _motion_snapshot(robot)
        for i, angle in enumerate(angles, 1):
            lower, upper = limits[f"joint{i}"]
            if not lower <= angle <= upper:
                raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다. 이동하지 않습니다.")
        if lows is None:
            lows, highs = angles.copy(), angles.copy()
        else:
            lows = [min(a, b) for a, b in zip(lows, angles)]
            highs = [max(a, b) for a, b in zip(highs, angles)]
        if any(upper - lower > tolerance for lower, upper in zip(lows, highs)):
            raise RuntimeError("이동 전 관찰 중 관절 각도가 0.05도 넘게 변했습니다. 새 목표를 보내지 않습니다.")
        if last_timestamp is None or timestamp > last_timestamp:
            new_samples += 1
            last_timestamp = timestamp
        if time.monotonic() - started >= 0.5:
            if new_samples < 3:
                raise RuntimeError("새 관절 피드백이 충분하지 않아 이동 전 검사를 완료하지 못했습니다.")
            if motion_status == 1:
                print("이전 목표 미도달 표시(1)가 있지만 관절 변화 검사를 통과했습니다.")
            leftover = robot.get_arm_status().msg.arm_status
            if int(leftover) in REJECTIONS:
                print("이전 경로 거부 표시({})가 남아 있습니다. 팔이 멈춰 있으므로 새 목표를 보낼 수 있습니다.".format(leftover))
            print("이동 전 검사 통과. 새 목표를 전송할 준비가 됐습니다.")
            return angles
        time.sleep(0.05)


def move_one_joint(robot, joint_index, degrees, timeout=10.0):
    if joint_index not in range(1, 8):
        raise ValueError("관절 번호는 1~7입니다.")
    if not math.isfinite(degrees) or not 0.5 <= abs(degrees) <= 5:
        raise ValueError("입문 예제는 한 번에 0.5~5도만 이동합니다. 음수도 가능합니다.")
    start = check_before_move(robot)
    target = start.copy()
    target[joint_index - 1] += math.radians(degrees)
    for i, angle in enumerate(target, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= angle <= high:
            raise RuntimeError(f"J{i} 목표가 SDK 범위 밖입니다. 이동하지 않습니다.")
    print("현재 각도(deg):", [round(math.degrees(a), 3) for a in start])
    print("목표 각도(deg):", [round(math.degrees(a), 3) for a in target])
    print(f"J{joint_index}를 현재보다 {degrees:+g}도 이동합니다. 속도 비율은 5%입니다.")
    return _execute_joint_target(robot, target, timeout)


def check_speed_percent(speed_percent):
    if isinstance(speed_percent, bool) or not isinstance(speed_percent, int) or not 1 <= speed_percent <= 100:
        raise ValueError("속도 비율은 1~100의 정수여야 합니다.")
    return speed_percent


class JointLine:
    """Targets a short lead ahead on the straight joint-space line to the goal.

    One move_j on firmware 1.12 turns every joint at the same rate, so joints
    with less travel arrive early and the flange wanders (2026-11-16 log).
    Re-sending move_j with a target that stays ``lead_degrees`` ahead of the
    slowest joint keeps all joints at the same fraction of their travel, so
    they arrive together. If the stream stops, the arm has at most the lead
    left to travel.
    """

    def __init__(self, start, target, lead_degrees):
        self.start, self.target = list(start), list(target)
        travel = [abs(t - s) for s, t in zip(start, target)]
        largest = max(travel)
        # Joints with little travel get targets only a hair ahead; pacing on
        # them could stall the stream, and they barely move the flange anyway.
        self.paced = [i for i, d in enumerate(travel) if d >= 0.25 * largest]
        self.lead = 1.0 if largest < math.radians(lead_degrees) else math.radians(lead_degrees) / largest
        self.fraction = 0.0

    def advance(self, angles):
        """The next target if the lead should move forward, otherwise None."""
        progress = 0.0 if self.lead >= 1.0 else min(
            (angles[i] - self.start[i]) / (self.target[i] - self.start[i]) for i in self.paced)
        wanted = min(1.0, max(0.0, progress) + self.lead)
        if wanted <= self.fraction:
            return None
        self.fraction = wanted
        if wanted >= 1.0:
            return list(self.target)
        return [s + (t - s) * wanted for s, t in zip(self.start, self.target)]


def move_joint_target(robot, target, timeout=30.0, *, max_step_degrees=5.0, on_feedback=None,
                      speed_percent=5, allow_long_move=False, lead_degrees=None, update_hz=100.0):
    """Execute a bounded seven-joint waypoint without enabling motors.

    ``allow_long_move`` lifts the 5-degree waypoint cap for a planned single
    move; ``max_step_degrees`` still bounds how far the target may be from the
    posture measured right before sending. With ``lead_degrees`` the joints are
    led along the straight joint-space line (see JointLine) instead of being
    given the final target at once; ``update_hz`` is how often that target is
    moved forward. At 20 Hz the real arm arrived together but ran unevenly
    (2026-11-16 22:52 log): between updates each joint reached its small target
    and paused, so the default refreshes five times as often in finer steps.
    """
    target = list(target)
    check_speed_percent(speed_percent)
    step_cap = 360 if allow_long_move else 5  # 360: no cap beyond the joint ranges themselves.
    if lead_degrees is not None and (not math.isfinite(lead_degrees) or not 0.5 <= lead_degrees <= 10):
        raise ValueError("관절 선행 각도는 0.5~10도입니다.")
    if not math.isfinite(update_hz) or not 10 <= update_hz <= 200:
        raise ValueError("관절 목표 갱신 빈도는 10~200Hz입니다.")
    if len(target) != 7 or not all(math.isfinite(a) for a in target):
        raise ValueError("목표 관절각은 유한한 rad 값 7개여야 합니다.")
    if not math.isfinite(max_step_degrees) or not 0 < max_step_degrees <= step_cap:
        raise ValueError("관절 웨이포인트 간격은 0보다 크고 {}도 이하여야 합니다.".format(step_cap))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("이동 대기 시간은 0보다 큰 유한한 초 값이어야 합니다.")
    for i, angle in enumerate(target, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= angle <= high:
            raise RuntimeError(f"J{i} 목표가 SDK 범위 밖입니다. 이동하지 않습니다.")
    start = check_before_move(robot)
    if max(abs(a - b) for a, b in zip(start, target)) > math.radians(max_step_degrees) + 1e-9:
        raise RuntimeError("현재 자세와 다음 관절 목표가 허용 간격보다 멉니다. 이동하지 않습니다.")
    return _execute_joint_target(robot, target, timeout, on_feedback=on_feedback, arrival_degrees=0.1,
                                 speed_percent=speed_percent, lead_degrees=lead_degrees,
                                 update_hz=update_hz)


def _execute_joint_target(robot, target, timeout, *, on_feedback=None, arrival_degrees=0.2,
                          speed_percent=5, lead_degrees=None, update_hz=100.0):
    require_planned_mode(robot)
    started = False
    watch = None
    try:
        if on_feedback is not None:
            on_feedback()
        start = motion_feedback(robot)[0]
        watch = RejectionWatch(robot, start)
        line = None if lead_degrees is None else JointLine(start, target, lead_degrees)
        robot.set_speed_percent(speed_percent)
        started = True
        last_sent = list(target) if line is None else line.advance(start)
        robot.move_j(last_sent)
        watch.target_sent()
        resend = ResendGuard([math.radians(0.05)])
        diagnose(robot, "joint_target_sent", target_rad=target, timeout_s=timeout,
                 lead_degrees=lead_degrees, update_hz=None if line is None else update_hz)
        print("move_j CAN 명령 전송 완료. 목표 도착을 확인합니다.")
        waiting = ProgressTimeout(timeout, [math.radians(0.1)])
        reached_samples = 0
        previous_stamps = None
        while True:
            if on_feedback is not None:
                on_feedback()
            angles, motion_status = motion_feedback(robot)
            watch.check(angles)
            ahead = None if line is None else line.advance(angles)
            if ahead is not None:
                last_sent = ahead
                robot.move_j(ahead)
                resend.sent()
            error = max(abs(a - b) for a, b in zip(angles, target))
            if error >= math.radians(arrival_degrees) and resend.due(motion_status == 0, [error]):
                robot.move_j(last_sent)
                resend.sent()
                diagnose(robot, "joint_target_resent", target_rad=last_sent, error_deg=math.degrees(error),
                         resend=resend.resends)
                print("컨트롤러가 도달을 보고했지만 관절이 목표에서 {:.2f}도 떨어져 있어 같은 목표를 다시 보냅니다 ({}회).".format(
                    math.degrees(error), resend.resends))
            parser = getattr(robot, "_parser", None)
            joint = robot.get_joint_angles()
            stamps = (joint.timestamp,) if parser is None else tuple(
                getattr(parser, name).timestamp for name in ("joint_12", "joint_34", "joint_56", "joint_7"))
            reached = motion_status == 0 and error < math.radians(arrival_degrees)
            if not reached:
                reached_samples = 0
            if previous_stamps is None or all(a > b for a, b in zip(stamps, previous_stamps)):
                reached_samples = reached_samples + 1 if reached else 0
                previous_stamps = stamps
            if reached_samples >= 3:
                print("목표 도착 확인. 자동 복귀/비활성화는 하지 않습니다.")
                return angles
            if waiting.expired([error]):
                raise RuntimeError(
                    f"이동 완료 확인 시간 초과({timeout:g}초 동안 목표에 가까워지지 않음): "
                    f"motion_status={int(motion_status)}, max_joint_error_deg={math.degrees(error):.3f}"
                )
            # While the target is still being led forward, refresh it in fine steps.
            time.sleep(0.05 if line is None or line.fraction >= 1.0 else 1.0 / update_hz)
    except BaseException as exc:
        if started:
            stop_failed_move(robot, exc, watch)
        raise


class JointPath:
    """Targets a short lead ahead of the arm along a joint-space polyline.

    Distance along the path is measured as the largest single-joint change, so
    ``lead_degrees`` means "the fastest-moving joint is this far ahead".
    Progress comes from projecting the measured joints onto the path and never
    moves backward. Like JointLine, but for a whole recorded path, so a replay
    is one continuous motion instead of a stop at every waypoint.
    """

    WINDOW = 8  # Segments searched ahead of the current one when projecting.

    def __init__(self, points, lead_degrees):
        self.points = [list(p) for p in points]
        self.lengths = [max(abs(b - a) for a, b in zip(p, q)) for p, q in zip(self.points, self.points[1:])]
        self.starts = [0.0]
        for length in self.lengths:
            self.starts.append(self.starts[-1] + length)
        self.total = self.starts[-1]
        self.lead = math.radians(lead_degrees)
        self.index, self.progress, self.sent = 0, 0.0, None

    @property
    def finished(self):
        return self.sent is not None and self.sent >= self.total

    def point_at(self, s):
        if s >= self.total:
            return list(self.points[-1])
        i = max(0, min(len(self.lengths) - 1, bisect.bisect_left(self.starts, s) - 1))
        length = self.lengths[i]
        u = 0.0 if length <= 0 else (s - self.starts[i]) / length
        return [a + (b - a) * u for a, b in zip(self.points[i], self.points[i + 1])]

    def locate(self, angles):
        """Advance progress to the measured joints; return their distance (rad) from the path."""
        best = None
        for i in range(self.index, min(len(self.lengths), self.index + self.WINDOW)):
            a, b = self.points[i], self.points[i + 1]
            span = [y - x for x, y in zip(a, b)]
            length2 = sum(d * d for d in span)
            u = 0.0 if length2 <= 0 else min(1.0, max(0.0, sum((q - x) * d for q, x, d in zip(angles, a, span)) / length2))
            near = [x + d * u for x, d in zip(a, span)]
            distance = max(abs(q - n) for q, n in zip(angles, near))
            if best is None or distance < best[0] - 1e-9:
                best = (distance, i, self.starts[i] + u * self.lengths[i])
        distance, index, s = best
        if s > self.progress:
            self.index, self.progress = index, s
        return distance

    def advance(self, angles):
        """(next target or None, distance of the arm from the path in rad)."""
        distance = self.locate(angles)
        wanted = min(self.total, self.progress + self.lead)
        if self.sent is not None and wanted <= self.sent + 1e-6:
            return None, distance
        self.sent = wanted
        return self.point_at(wanted), distance


def follow_joint_path(robot, points, *, timeout=30.0, speed_percent=5, lead_degrees=2.0, update_hz=100.0,
                      max_deviation_degrees=5.0, arrival_degrees=0.1, on_feedback=None):
    """Drive the joints continuously along ``points`` (rad) and stop at the last one.

    The arm must start within 0.5 degree of the first point. Targets stay
    ``lead_degrees`` ahead along the path and are refreshed at ``update_hz``.
    Fails, with the usual stop policy, if the arm strays more than
    ``max_deviation_degrees`` from the path or stops making progress for
    ``timeout`` seconds. Returns (final angles, largest distance from the path in deg).
    """
    points = [list(p) for p in points]
    check_speed_percent(speed_percent)
    if len(points) < 2 or any(len(p) != 7 or not all(math.isfinite(a) for a in p) for p in points):
        raise ValueError("경로는 유한한 rad 관절각 7개짜리 점 2개 이상이어야 합니다.")
    if not math.isfinite(lead_degrees) or not 0.5 <= lead_degrees <= 10:
        raise ValueError("관절 선행 각도는 0.5~10도입니다.")
    if not math.isfinite(update_hz) or not 10 <= update_hz <= 200:
        raise ValueError("관절 목표 갱신 빈도는 10~200Hz입니다.")
    limits = robot.get_config()["joint_limits"]
    for point in points:
        for i, angle in enumerate(point, 1):
            low, high = limits[f"joint{i}"]
            if not low <= angle <= high:
                raise RuntimeError(f"경로의 J{i} 각도가 SDK 범위 밖입니다. 이동하지 않습니다.")
    start = check_before_move(robot)
    if max(abs(a - b) for a, b in zip(start, points[0])) > math.radians(0.5):
        raise RuntimeError("현재 자세가 경로의 시작점에서 0.5도 넘게 벗어났습니다. 이동하지 않습니다.")
    require_planned_mode(robot)
    path = JointPath(points, lead_degrees)
    final = points[-1]
    started = False
    watch = None
    worst = 0.0
    try:
        if on_feedback is not None:
            on_feedback()
        watch = RejectionWatch(robot, start)
        robot.set_speed_percent(speed_percent)
        started = True
        last_sent, _ = path.advance(start)
        robot.move_j(last_sent)
        watch.target_sent()
        resend = ResendGuard([math.radians(0.05)])
        diagnose(robot, "joint_path_started", points=len(points), length_deg=math.degrees(path.total),
                 lead_degrees=lead_degrees, update_hz=update_hz)
        print("경로를 연속으로 따라갑니다: 점 {}개, 가장 많이 도는 관절 기준 경로 길이 {:.1f}도.".format(
            len(points), math.degrees(path.total)))
        waiting = ProgressTimeout(timeout, [math.radians(0.1), math.radians(0.05)])
        reached_samples = 0
        previous_stamps = None
        while True:
            if on_feedback is not None:
                on_feedback()
            angles, motion_status = motion_feedback(robot)
            watch.check(angles)
            ahead, distance = path.advance(angles)
            worst = max(worst, distance)
            if distance > math.radians(max_deviation_degrees):
                raise RuntimeError("관절이 기록 경로에서 {:.1f}도 벗어났습니다 (한도 {:g}도).".format(
                    math.degrees(distance), max_deviation_degrees))
            if ahead is not None:
                last_sent = ahead
                robot.move_j(ahead)
                resend.sent()
            error = max(abs(a - b) for a, b in zip(angles, final))
            if (path.finished and error >= math.radians(arrival_degrees)
                    and resend.due(motion_status == 0, [error])):
                robot.move_j(last_sent)
                resend.sent()
                diagnose(robot, "joint_target_resent", target_rad=last_sent, error_deg=math.degrees(error),
                         resend=resend.resends)
                print("끝점 도달 보고 후에도 {:.2f}도 떨어져 있어 끝점을 다시 보냅니다 ({}회).".format(
                    math.degrees(error), resend.resends))
            parser = getattr(robot, "_parser", None)
            joint = robot.get_joint_angles()
            stamps = (joint.timestamp,) if parser is None else tuple(
                getattr(parser, name).timestamp for name in ("joint_12", "joint_34", "joint_56", "joint_7"))
            reached = path.finished and motion_status == 0 and error < math.radians(arrival_degrees)
            if not reached:
                reached_samples = 0
            if previous_stamps is None or all(a > b for a, b in zip(stamps, previous_stamps)):
                reached_samples = reached_samples + 1 if reached else 0
                previous_stamps = stamps
            if reached_samples >= 3:
                diagnose(robot, "joint_path_completed", max_deviation_deg=math.degrees(worst))
                print("경로 끝점 도착 확인. 경로에서 가장 멀리 벗어난 정도 {:.2f}도.".format(math.degrees(worst)))
                return angles, math.degrees(worst)
            if waiting.expired([path.total - path.progress, error]):
                raise RuntimeError("경로 진행 확인 시간 초과({:g}초 동안 진행 없음): 남은 경로 {:.2f}도, 끝점 오차 {:.3f}도".format(
                    timeout, math.degrees(path.total - path.progress), math.degrees(error)))
            time.sleep(0.05 if path.finished else 1.0 / update_hz)
    except BaseException as exc:
        if started:
            stop_failed_move(robot, exc, watch)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--joint", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--degrees", type=float, required=True, help="현재 각도에 더할 도수. 절댓값 0.5~5")
    args = parser.parse_args()
    robot = None
    try:
        if not math.isfinite(args.degrees) or not 0.5 <= abs(args.degrees) <= 5:
            raise ValueError("이동 각도는 -5~-0.5 또는 0.5~5도입니다.")
        print("실제 이동을 실행합니다. 팔과 도구가 움직일 공간을 비우세요.")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        move_one_joint(robot, args.joint, args.degrees)
        print_status(robot)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("이동 중단:", str(exc) or "사용자 중단")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
