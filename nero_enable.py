"""Beginner step 2: enable Nero motors and verify CAN control; no move targets."""

import argparse
import math
import time

from nero_read_status import connect_nero, print_status, recent


DRIVER_ERROR_FIELDS = (
    "voltage_too_low", "motor_overheating", "driver_overcurrent",
    "driver_overheating", "collision_status", "driver_error_status", "stall_status",
)


def check_before_enable(robot, *, allow_disabled_not_reached=False, allow_enabled_not_reached=False):
    """Check enable state without sending targets or clearing an old target flag."""
    joint = robot.get_joint_angles()
    status = robot.get_arm_status()
    if not recent(joint) or not recent(status):
        raise RuntimeError("최근 관절/상태 피드백이 없습니다.")
    if status.msg.arm_status == 1:
        raise RuntimeError(
            "현재 EMERGENCY_STOP(급정지) 상태입니다. 모터 활성화와 급정지 해제는 별개이며 "
            "enable로 해제하지 않습니다. 급정지 원인과 팔 지지 상태를 확인하세요. "
            "nero_reset.py는 기본 조회만 하며 --execute를 지정해야 복구를 요청합니다."
        )
    if status.msg.arm_status not in (0, 6):
        raise RuntimeError(f"활성화할 수 있는 상태가 아닙니다. arm_status={status.msg.arm_status}")
    arm_status = status.msg.arm_status
    motion_status = status.msg.motion_status
    angles = list(joint.msg)
    if len(angles) != 7:
        raise RuntimeError("7개 관절 각도를 확인하지 못했습니다.")
    all_disabled = True
    all_enabled = True
    for i, angle in enumerate(angles, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not math.isfinite(angle) or not low <= angle <= high:
            raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다. 활성화하지 않습니다.")
        driver = robot.get_driver_states(i)
        if not recent(driver):
            raise RuntimeError(f"J{i}의 최근 드라이버 피드백이 없습니다.")
        foc = driver.msg.foc_status
        all_disabled = all_disabled and not foc.driver_enable_status
        all_enabled = all_enabled and foc.driver_enable_status
        if any(getattr(foc, name) for name in DRIVER_ERROR_FIELDS):
            raise RuntimeError(f"J{i} 드라이버에 오류가 보고됐습니다.")
    disabled_not_reached = (
        allow_disabled_not_reached and all_disabled
        and arm_status == 6 and motion_status == 1
    )
    enabled_not_reached = (
        allow_enabled_not_reached and all_enabled and arm_status == 0
        and status.msg.ctrl_mode == 1 and motion_status == 1
    )
    if motion_status != 0 and not (disabled_not_reached or enabled_not_reached):
        raise RuntimeError("목표 도달 상태가 아닙니다. 기존 명령/제어 상태를 확인하세요.")
    return disabled_not_reached


def enable_only(robot):
    deadline = time.monotonic() + 5.0
    while True:
        robot.enable()
        status = robot.get_arm_status()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if (recent(status) and status.msg.ctrl_mode == 1
                and status.msg.arm_status == 0
                and all(recent(d) and d.msg.foc_status.driver_enable_status for d in drivers)):
            return
        if recent(status) and status.msg.arm_status not in (0, 6):
            raise RuntimeError("활성화 중 로봇 오류가 보고됐습니다.")
        if time.monotonic() >= deadline:
            raise RuntimeError("활성화/CAN 제어/정상 상태를 제한 시간 내 확인하지 못했습니다.")
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    args = parser.parse_args()
    robot = None
    attempted = False
    try:
        print("2단계: 모터 활성화 전 상태를 확인합니다. 브레이크 해제 시 자세가 달라질 수 있습니다.")
        print("이동 목표는 보내지 않습니다. 주변을 비우고 다른 제어 프로그램의 동작을 중지하세요.")
        robot, version, profile = connect_nero(args.channel)
        print(f"펌웨어: {version} / SDK 드라이버: {profile}")
        time.sleep(1.0)
        if check_before_enable(robot, allow_disabled_not_reached=True, allow_enabled_not_reached=True):
            print("모터 7개 비활성화/브레이크 미해제 상태에서 목표 미도달 표시가 있습니다.")
            print("이 표시만으로 활성화를 막지 않고 enable을 요청합니다. reset/이동 목표는 보내지 않습니다.")
        status = robot.get_arm_status()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if (recent(status) and status.msg.ctrl_mode == 1 and status.msg.arm_status == 0
                and all(recent(d) and d.msg.foc_status.driver_enable_status for d in drivers)):
            print("이미 모터 7개가 활성화된 정상 CAN 제어 상태입니다. enable을 다시 보내지 않습니다.")
        else:
            print("사전 검사 통과. 모터 활성화 요청을 보냅니다.")
            attempted = True
            enable_only(robot)
        print_status(robot)
        print("2단계 완료: 모터 7개 활성화 / CAN 제어 / 정상 상태를 확인했습니다.")
        status = robot.get_arm_status()
        if recent(status) and status.msg.motion_status != 0:
            print("이전 목표 미도달 표시는 남아 있습니다. 이동 프로그램에서 관절 변화를 관찰한 뒤 새 목표를 보냅니다.")
        print("프로그램을 종료해도 모터는 활성화 상태로 남습니다.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("2단계 중단:", str(exc) or "사용자 중단")
        if attempted:
            try:
                robot.electronic_emergency_stop()
                print("전자 급정지 명령을 보냈습니다. 실제 정지 여부와 상태를 확인하세요.")
            except Exception as stop_error:
                print("전자 급정지 전송 실패:", stop_error)
                print("장비의 물리 비상정지 절차를 따르세요.")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
