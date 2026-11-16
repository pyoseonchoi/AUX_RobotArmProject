"""Beginner step 4: translate the flange 1 to 5 mm along one base-frame axis."""

import argparse
import math
import time

from pyAgxArm.utiles.tf import euler_convert_quat
from nero_enable import recent
from nero_move_joint import (ProgressTimeout, RejectionWatch, check_before_move, check_speed_percent,
                             motion_feedback, require_planned_mode, stop_failed_move,
                             wait_for_motion_feedback)
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, print_status, require_feedback, read_flange_pose
from nero_diagnostics import diagnose


def flange_pose(robot, *, newer_than=None):
    return read_flange_pose(robot, newer_than=newer_than)


def orientation_error(actual, target):
    # Quaternion distance treats +/-pi Euler wraparound as the same orientation.
    q_actual = euler_convert_quat(*actual[3:])
    q_target = euler_convert_quat(*target[3:])
    dot = abs(sum(a * b for a, b in zip(q_actual, q_target)))
    return 2 * math.acos(min(1.0, max(0.0, dot)))


def move_linear_offset(robot, axis, millimeters, timeout=10.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("이동 대기 시간은 0보다 큰 유한한 초 값이어야 합니다.")
    if axis not in ("x", "y", "z"):
        raise ValueError("이동 축은 x, y, z 중 하나입니다.")
    if not math.isfinite(millimeters) or not 1 <= abs(millimeters) <= 5:
        raise ValueError("입문 예제는 한 번에 절댓값 1~5mm만 이동합니다.")
    check_before_move(robot)
    start, _ = flange_pose(robot)
    target = start.copy()
    target[("x", "y", "z").index(axis)] += millimeters / 1000.0
    print(f"베이스 {axis.upper()} 방향으로 {millimeters:+g}mm 직선 이동합니다.")
    return _execute_linear_target(robot, target, timeout)


def move_linear_target(robot, target, timeout=30.0, *, on_feedback=None, speed_percent=5, settled=False):
    """Execute an absolute flange pose (m/rad), retaining feedback checks.

    ``settled``: the caller has just confirmed, on fresh feedback, that a move
    of its own ended with the arm at rest (move_joint_target returned). The
    0.5 s stillness window is then skipped, which was most of the pause between
    the two parts of a planned move (about 0.85 s on 2026-11-16). Fresh
    feedback, motor, fault, mode and joint-limit checks still run.
    """
    target = list(target)
    check_speed_percent(speed_percent)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("이동 대기 시간은 0보다 큰 유한한 초 값이어야 합니다.")
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError("목표 위치/방향은 유한한 숫자 6개여야 합니다.")
    if (abs(target[3]) > math.pi or abs(target[4]) > math.pi / 2
            or abs(target[5]) > math.pi):
        raise ValueError("목표 방향이 SDK의 RPY 범위를 벗어났습니다.")
    if settled:
        angles, _, _ = wait_for_motion_feedback(robot, newer_than=feedback_time())
        for i, angle in enumerate(angles, 1):
            lower, upper = robot.get_config()["joint_limits"][f"joint{i}"]
            if not lower <= angle <= upper:
                raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다. 이동하지 않습니다.")
    else:
        check_before_move(robot)
    return _execute_linear_target(robot, target, timeout, on_feedback=on_feedback,
                                  speed_percent=speed_percent)


def _execute_linear_target(robot, target, timeout, *, on_feedback=None, speed_percent=5):
    """Send a target after the public entry point has observed joint feedback."""
    watch = RejectionWatch(robot, motion_feedback(robot)[0])
    require_planned_mode(robot)
    start, _ = flange_pose(robot)
    print("현재 위치(mm):", [round(value * 1000, 3) for value in start[:3]])
    print("목표 위치(mm):", [round(value * 1000, 3) for value in target[:3]])
    print("목표 방향(deg):", [round(math.degrees(value), 3) for value in target[3:]])
    print("속도 비율은 {}%입니다.".format(speed_percent))
    started = False
    try:
        if on_feedback is not None:
            on_feedback()
        robot.set_speed_percent(speed_percent)
        started = True
        sent_at = feedback_time()
        robot.move_l(target.copy())
        watch.target_sent()
        diagnose(robot, "linear_target_sent", target_m_rad=target, timeout_s=timeout)
        print("move_l CAN 명령 전송 완료. 컨트롤러 상태와 목표 위치 오차를 확인합니다.")
        waiting = ProgressTimeout(timeout, [0.0005, math.radians(0.5)])
        reached_samples = 0
        previous_stamps = None
        last_log = -float("inf")
        while True:
            if on_feedback is not None:
                on_feedback()
            joint_angles, motion_status = motion_feedback(robot)
            watch.check(joint_angles)
            actual, timestamp = flange_pose(robot)
            position_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual[:3], target[:3])))
            angle_error = orientation_error(actual, target)
            status = robot.get_arm_status()
            parser = getattr(robot, "_parser", None)
            stamps = (timestamp,) if parser is None else tuple(getattr(parser, name).timestamp
                for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"))
            reached = (motion_status == 0 and position_error < 0.0005
                       and angle_error < math.radians(1) and int(status.msg.mode_feedback) == 2
                       and status.timestamp > sent_at and all(t > sent_at for t in stamps))
            if not reached:
                reached_samples = 0
            if previous_stamps is None or all(a > b for a, b in zip(stamps, previous_stamps)):
                reached_samples = reached_samples + 1 if reached else 0
                previous_stamps = stamps
            if time.monotonic() - last_log >= 0.5:
                diagnose(robot, "linear_progress", position_error_mm=position_error * 1000,
                         orientation_error_deg=math.degrees(angle_error))
                last_log = time.monotonic()
            if reached_samples >= 3:
                diagnose(robot, "linear_arrived", position_error_mm=position_error * 1000)
                print("직선 이동 도착을 확인했습니다.")
                print(f"위치 오차: {position_error * 1000:.3f} mm")
                print("도착 관절 각도(deg):", [round(math.degrees(value), 3) for value in joint_angles])
                print("자동 복귀/비활성화는 하지 않습니다.")
                return actual
            if waiting.expired([position_error, angle_error]):
                status = robot.get_arm_status()
                mode = status.msg.mode_feedback if recent(status) else "unknown"
                raise RuntimeError(
                    f"직선 이동 완료 확인 시간 초과({timeout:g}초 동안 목표에 가까워지지 않음): "
                    f"motion_status={int(motion_status)}, "
                    f"mode_feedback={mode}, position_error_mm={position_error * 1000:.3f}, "
                    f"orientation_error_deg={math.degrees(angle_error):.3f}"
                )
            time.sleep(0.05)
    except BaseException as exc:
        if started:
            stop_failed_move(robot, exc, watch)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--mm", type=float, required=True, help="현재 위치에 더할 mm. 절댓값 1~5")
    args = parser.parse_args()
    robot = None
    try:
        if not math.isfinite(args.mm) or not 1 <= abs(args.mm) <= 5:
            raise ValueError("이동 거리는 -5~-1 또는 1~5mm입니다.")
        print("실제 직선 이동을 실행합니다. 팔과 도구가 움직일 공간을 비우세요.")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        move_linear_offset(robot, args.axis, args.mm)
        print_status(robot)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("직선 이동 중단:", str(exc) or "사용자 중단")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
