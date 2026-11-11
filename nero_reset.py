"""Inspect emergency-stop state; --execute requests reset once, without enable/move."""

import argparse
import math
import time

from nero_enable import DRIVER_ERROR_FIELDS, recent
from nero_read_status import connect_nero, feedback_time, print_status


def reset_feedback(robot):
    status = robot.get_arm_status()
    joints = robot.get_joint_angles()
    if not recent(status) or not recent(joints):
        raise RuntimeError("급정지 복구를 판단할 최근 상태/관절 피드백이 없습니다.")
    if status.msg.arm_status != 1 or status.msg.ctrl_mode != 1:
        raise RuntimeError(f"급정지 CAN 제어 상태가 아닙니다: {status.msg}")
    values = list(joints.msg)
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("관절 각도 7개를 확인하지 못했습니다.")
    for i, value in enumerate(values, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= value <= high:
            raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다. 제조사 복구 절차를 확인하세요.")
        driver = robot.get_driver_states(i)
        if not recent(driver) or not driver.msg.foc_status.driver_enable_status:
            raise RuntimeError(f"J{i} 모터 활성화를 확인하지 못했습니다. 이 파일은 자동 enable하지 않습니다.")
        if any(getattr(driver.msg.foc_status, name) for name in DRIVER_ERROR_FIELDS):
            raise RuntimeError(f"J{i} 드라이버 오류가 보고됐습니다. reset하지 않습니다.")
    return values


def reset_emergency_stop(robot, timeout=3.0):
    # Observe a supported arm before changing stop state. This is not a
    # physical support detector; the operator must secure the arm/tool.
    low = high = None
    started = time.monotonic()
    while True:
        values = reset_feedback(robot)
        if low is None:
            low, high = values.copy(), values.copy()
        else:
            low = [min(a, b) for a, b in zip(low, values)]
            high = [max(a, b) for a, b in zip(high, values)]
        if any(b - a > math.radians(0.05) for a, b in zip(low, high)):
            raise RuntimeError("관절 각도가 변하고 있습니다. 팔/도구의 지지를 확인하세요. reset하지 않습니다.")
        if time.monotonic() - started >= 0.5:
            break
        time.sleep(0.05)
    sent_at = feedback_time()
    robot.reset()
    print("reset 요청을 1회 전송했습니다. enable/이동 명령은 보내지 않습니다.")
    deadline = time.monotonic() + timeout
    while True:
        status = robot.get_arm_status()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if (recent(status) and status.timestamp > sent_at
                and all(recent(d) and d.timestamp > sent_at for d in drivers)):
            if status.msg.arm_status in (0, 6):
                print("급정지 상태에서 벗어난 피드백을 확인했습니다.")
                return
            if status.msg.arm_status != 1:
                raise RuntimeError(f"reset 후 다른 상태가 보고됐습니다: {status.msg}")
        if time.monotonic() >= deadline:
            raise RuntimeError("reset 후 급정지 해제를 확인하지 못했습니다. 자동 재시도하지 않습니다.")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--execute", action="store_true", help="팔/도구를 안전하게 지지한 뒤 급정지 복구를 1회 요청")
    args = parser.parse_args()
    robot = None
    try:
        print("제조사 문서: reset 시 모터가 비활성화되어 팔이 처질 수 있습니다.")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        print_status(robot)
        if not args.execute:
            print("조회만 완료했습니다. 복구 명령은 보내지 않았습니다.")
            return 0
        print("팔과 도구가 안전하게 지지된 상태에서 사용하는 복구 요청입니다.")
        reset_emergency_stop(robot)
        print_status(robot)
        print("자동 활성화/이동은 하지 않습니다. 출력된 상태를 확인하세요.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("급정지 복구 확인 중단:", str(exc) or "사용자 중단")
        print("원인 확인 없이 반복 reset하지 마세요.")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
