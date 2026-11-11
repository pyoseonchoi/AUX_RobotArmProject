"""Beginner step 4: translate the flange 1 to 5 mm along one base-frame axis."""

import argparse
import math
import time

from pyAgxArm.utiles.tf import euler_convert_quat
from nero_enable import recent
from nero_move_joint import check_before_move, motion_feedback
from nero_read_status import FeedbackUnavailableError, connect_nero, print_status, require_feedback


def flange_pose(robot, *, newer_than=None):
    feedback = robot.get_flange_pose()
    required = [("플랜지 통합값", feedback)]
    parser = getattr(robot, "_parser", None)
    if parser is not None:
        required.extend((name, getattr(parser, attr, None)) for name, attr in (
            ("플랜지 X/Y", "end_pose_xy"), ("플랜지 Z/roll", "end_pose_zrx"),
            ("플랜지 pitch/yaw", "end_pose_ryrz"),
        ))
    require_feedback(required, newer_than=newer_than)
    pose = list(feedback.msg)
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise RuntimeError("플랜지 위치/자세 6개 값을 확인하지 못했습니다.")
    return pose, feedback.timestamp


def orientation_error(actual, target):
    # Quaternion distance treats +/-pi Euler wraparound as the same orientation.
    q_actual = euler_convert_quat(*actual[3:])
    q_target = euler_convert_quat(*target[3:])
    dot = abs(sum(a * b for a, b in zip(q_actual, q_target)))
    return 2 * math.acos(min(1.0, max(0.0, dot)))


def move_linear_offset(robot, axis, millimeters, timeout=10.0):
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


def move_linear_target(robot, target, timeout=10.0):
    """Execute an absolute flange pose (m/rad), retaining feedback checks."""
    target = list(target)
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError("목표 위치/방향은 유한한 숫자 6개여야 합니다.")
    if (abs(target[3]) > math.pi or abs(target[4]) > math.pi / 2
            or abs(target[5]) > math.pi):
        raise ValueError("목표 방향이 SDK의 RPY 범위를 벗어났습니다.")
    check_before_move(robot)
    return _execute_linear_target(robot, target, timeout)


def _execute_linear_target(robot, target, timeout):
    """Send a target after the public entry point has observed joint feedback."""
    motion_feedback(robot)
    start, _ = flange_pose(robot)
    print("현재 위치(mm):", [round(value * 1000, 3) for value in start[:3]])
    print("목표 위치(mm):", [round(value * 1000, 3) for value in target[:3]])
    print("목표 방향(deg):", [round(math.degrees(value), 3) for value in target[3:]])
    print("속도 비율은 5%입니다.")
    started = False
    try:
        robot.set_speed_percent(5)
        started = True
        robot.move_l(target.copy())
        print("move_l CAN 명령 전송 완료. 컨트롤러 상태와 목표 위치 오차를 확인합니다.")
        deadline = time.monotonic() + timeout
        reached_samples = 0
        last_timestamp = None
        while True:
            joint_angles, motion_status = motion_feedback(robot)
            actual, timestamp = flange_pose(robot)
            position_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual[:3], target[:3])))
            angle_error = orientation_error(actual, target)
            if timestamp != last_timestamp:
                reached = motion_status == 0 and position_error < 0.0005 and angle_error < math.radians(1)
                reached_samples = reached_samples + 1 if reached else 0
                last_timestamp = timestamp
            if reached_samples >= 3:
                print("직선 이동 도착을 확인했습니다.")
                print(f"위치 오차: {position_error * 1000:.3f} mm")
                print("도착 관절 각도(deg):", [round(math.degrees(value), 3) for value in joint_angles])
                print("자동 복귀/비활성화는 하지 않습니다.")
                return actual
            if time.monotonic() >= deadline:
                status = robot.get_arm_status()
                mode = status.msg.mode_feedback if recent(status) else "unknown"
                raise RuntimeError(
                    f"직선 이동 완료 확인 시간 초과: motion_status={int(motion_status)}, "
                    f"mode_feedback={mode}, position_error_mm={position_error * 1000:.3f}, "
                    f"orientation_error_deg={math.degrees(angle_error):.3f}"
                )
            time.sleep(0.05)
    except BaseException as exc:
        if started:
            print("급정지 요청 원인:", type(exc).__name__, str(exc) or "사용자 중단")
            try:
                robot.electronic_emergency_stop()
                print("전자 급정지를 요청했습니다. 실제 정지 여부를 확인하세요.")
            except Exception as exc:
                print("전자 급정지 전송 실패:", exc)
                print("장비의 물리 비상정지 절차를 따르세요.")
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
