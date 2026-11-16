"""Inspect emergency-stop state; --execute requests reset once, without enable/move."""

import argparse
import math
import time

from nero_enable import DRIVER_ERROR_FIELDS
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, print_status, require_feedback
from nero_diagnostics import SessionDiagnostics, diagnose


def reset_hint(status, enabled):
    if int(status.arm_status) == 1:
        if int(status.ctrl_mode) != 1:
            return ("급정지 상태지만 CAN_CTRL이 아닙니다. 이 복구 도구는 급정지 중 제어 모드를 강제로 바꾸지 않습니다. "
                    "팔을 지지하고 제조사 복구 절차로 확인하세요.")
        if not all(enabled):
            return ("급정지 상태지만 모터 7개가 모두 활성 상태가 아닙니다. 제조사 reset 적용 조건과 다릅니다. "
                    "자동 enable/reset은 하지 않습니다. 팔을 지지하고 제조사 복구 절차로 확인하세요.")
        return "전자 급정지 상태입니다. 팔/도구를 안전하게 지지한 뒤 --execute로 reset을 한 번 요청할 수 있습니다."
    if int(status.ctrl_mode) == 6:
        return ("급정지가 아니라 교시 제어 상태입니다. 팔을 지지하고 "
                "python nero_teach.py control --firmware v112 --execute로 복귀하세요.")
    if int(status.arm_status) in (0, 6) and int(status.ctrl_mode) == 1:
        return "급정지가 아닙니다. reset은 필요하지 않습니다. 비활성 모터는 별도의 enable 단계로 준비하세요."
    return "일반 전자 급정지 복구 대상이 아닙니다. 표시된 제어 모드와 오류 상태를 확인하세요."


def _reset_snapshot(robot, *, newer_than=None):
    status, joints = robot.get_arm_status(), robot.get_joint_angles()
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    required = [("로봇 상태", status), ("관절 통합값", joints)]
    required += [("J{} 드라이버".format(i), d) for i, d in enumerate(drivers, 1)]
    parser = getattr(robot, "_parser", None)
    if parser is not None:
        required += [(name, getattr(parser, name, None))
                     for name in ("joint_12", "joint_34", "joint_56", "joint_7")]
    require_feedback(required, newer_than=newer_than)
    for i, driver in enumerate(drivers, 1):
        faults = [name for name in DRIVER_ERROR_FIELDS if getattr(driver.msg.foc_status, name)]
        if faults:
            raise RuntimeError("J{} 드라이버 오류: {}. reset으로 해결됐다고 판단하지 않습니다.".format(i, faults))
    return status, joints, drivers


def reset_feedback(robot):
    status, joints, drivers = _reset_snapshot(robot)
    enabled = [bool(d.msg.foc_status.driver_enable_status) for d in drivers]
    if status.msg.arm_status != 1 or status.msg.ctrl_mode != 1:
        raise RuntimeError(reset_hint(status.msg, enabled) + " ctrl_mode={}".format(status.msg.ctrl_mode))
    if not all(enabled):
        raise RuntimeError(reset_hint(status.msg, enabled))
    values = list(joints.msg)
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("관절 각도 7개를 확인하지 못했습니다.")
    for i, value in enumerate(values, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= value <= high:
            raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다. 제조사 복구 절차를 확인하세요.")
    return values


def reset_emergency_stop(robot, timeout=3.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout은 0보다 큰 유한한 초 값이어야 합니다.")
    ready_deadline = time.monotonic() + 2.0
    requested_at = feedback_time()
    while True:
        try:
            _reset_snapshot(robot, newer_than=requested_at)
            break
        except FeedbackUnavailableError:
            if time.monotonic() >= ready_deadline:
                raise
            time.sleep(0.02)
    # Observe a supported arm before changing stop state. This is not a
    # physical support detector; the operator must secure the arm/tool.
    low = high = None
    previous = None
    fresh_count = 0
    started = time.monotonic()
    while True:
        values = reset_feedback(robot)
        parser = getattr(robot, "_parser", None)
        stamps = ((robot.get_joint_angles().timestamp,) if parser is None else tuple(
            getattr(parser, name).timestamp for name in ("joint_12", "joint_34", "joint_56", "joint_7")))
        if previous is None or all(a > b for a, b in zip(stamps, previous)):
            fresh_count += 1
            previous = stamps
        if low is None:
            low, high = values.copy(), values.copy()
        else:
            low = [min(a, b) for a, b in zip(low, values)]
            high = [max(a, b) for a, b in zip(high, values)]
        if any(b - a > math.radians(0.05) for a, b in zip(low, high)):
            raise RuntimeError("관절 각도가 변하고 있습니다. 팔/도구의 지지를 확인하세요. reset하지 않습니다.")
        if time.monotonic() - started >= 0.5:
            if fresh_count < 3:
                raise RuntimeError("새 관절 피드백이 충분하지 않아 reset 전 정지를 확인하지 못했습니다.")
            break
        time.sleep(0.05)
    sent_at = feedback_time()
    robot.reset()
    diagnose(robot, "reset_sent")
    print("reset 요청을 1회 전송했습니다. enable/이동 명령은 보내지 않습니다.")
    deadline = time.monotonic() + timeout
    last_reason = "새 피드백 대기"
    while True:
        try:
            status, _, drivers = _reset_snapshot(robot, newer_than=sent_at)
            enabled = [bool(d.msg.foc_status.driver_enable_status) for d in drivers]
            if status.msg.arm_status in (0, 6) and not any(enabled):
                print("급정지 해제와 모터 7개 비활성을 확인했습니다.")
                diagnose(robot, "reset_verified")
                return
            if status.msg.arm_status not in (0, 1, 6):
                raise RuntimeError(f"reset 후 다른 상태가 보고됐습니다: {status.msg}")
            last_reason = "arm_status={} / enabled={}".format(status.msg.arm_status, enabled)
        except FeedbackUnavailableError as exc:
            last_reason = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError("reset 후 급정지 해제/모터 비활성을 확인하지 못했습니다: {}. 자동 재시도하지 않습니다.".format(last_reason))
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--execute", action="store_true", help="팔/도구를 안전하게 지지한 뒤 급정지 복구를 1회 요청")
    parser.add_argument("--timeout", type=float, default=3.0, help="reset 응답 대기 시간(초), 기본 3")
    args = parser.parse_args()
    robot = None
    diagnostics = None
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout은 0보다 큰 유한한 초 값이어야 합니다.")
        diagnostics = SessionDiagnostics()
        print("실행 진단 로그:", diagnostics.path.resolve())
        diagnostics.event(None, "start", command="reset", options=vars(args))
        print("제조사 문서: reset 시 모터가 비활성화되어 팔이 처질 수 있습니다.")
        robot, version, profile = connect_nero(args.channel)
        robot._nero_diagnostics = diagnostics
        time.sleep(1.0)
        diagnostics.event(robot, "connected", firmware=version, sdk_profile=str(profile))
        print_status(robot)
        status, _, drivers = _reset_snapshot(robot)
        print(reset_hint(status.msg, [bool(d.msg.foc_status.driver_enable_status) for d in drivers]))
        if not args.execute:
            diagnostics.event(robot, "inspection_completed")
            print("조회만 완료했습니다. 복구 명령은 보내지 않았습니다.")
            return 0
        if status.msg.arm_status in (0, 6) and status.msg.ctrl_mode == 1:
            print("이미 급정지가 해제된 상태이므로 reset 명령을 보내지 않았습니다.")
            diagnostics.event(robot, "reset_not_needed")
            return 0
        print("팔과 도구가 안전하게 지지된 상태에서 사용하는 복구 요청입니다.")
        reset_emergency_stop(robot, timeout=args.timeout)
        print_status(robot)
        print("자동 활성화/이동은 하지 않습니다. 출력된 상태를 확인하세요.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=str(exc) or "사용자 중단")
        print("급정지 복구 확인 중단:", str(exc) or "사용자 중단")
        print("원인 확인 없이 반복 reset하지 마세요.")
        return 1
    finally:
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            if diagnostics is not None:
                diagnostics.close()


if __name__ == "__main__":
    raise SystemExit(main())
