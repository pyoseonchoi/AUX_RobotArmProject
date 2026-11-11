"""Beginner step 3: move one already-enabled joint by 0.5 to 5 degrees."""

import argparse
import math
import time

from nero_enable import DRIVER_ERROR_FIELDS, recent
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, print_status, require_feedback


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
    if status.msg.ctrl_mode != 1 or status.msg.arm_status != 0:
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
    started = False
    try:
        robot.set_speed_percent(5)
        started = True
        robot.move_j(target)
        print("move_j CAN 명령 전송 완료. 목표 도착을 확인합니다.")
        deadline = time.monotonic() + timeout
        reached_samples = 0
        while True:
            angles, motion_status = motion_feedback(robot)
            error = max(abs(a - b) for a, b in zip(angles, target))
            reached_samples = reached_samples + 1 if motion_status == 0 and error < math.radians(0.2) else 0
            if reached_samples >= 3:
                print("목표 도착 확인. 자동 복귀/비활성화는 하지 않습니다.")
                return angles
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"이동 완료 확인 시간 초과: motion_status={int(motion_status)}, "
                    f"max_joint_error_deg={math.degrees(error):.3f}"
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
