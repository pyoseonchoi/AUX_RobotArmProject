"""Request motor enable and verify all seven motors; never send motion/reset."""

import argparse
import math
import time

from nero_read_status import connect_nero, feedback_time, print_status, recent
from nero_diagnostics import SessionDiagnostics


DRIVER_ERROR_FIELDS = (
    "voltage_too_low", "motor_overheating", "driver_overcurrent",
    "driver_overheating", "collision_status", "driver_error_status", "stall_status",
)


def check_before_enable(robot, *, allow_disabled_not_reached=False, allow_enabled_not_reached=False):
    """Legacy strict check for callers that need it; the enable CLI does not use it."""
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


def enable_only(robot, timeout=5.0):
    """Enable independently of old planning errors; require new driver feedback."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("활성화 대기 시간은 0보다 큰 유한한 값이어야 합니다.")
    deadline = time.monotonic() + timeout
    requested_at = feedback_time()
    while True:
        status = robot.get_arm_status()
        if recent(status) and status.msg.arm_status == 1:
            raise RuntimeError("EMERGENCY_STOP 상태입니다. enable은 급정지를 해제하지 않습니다. nero_reset.py로 상태를 확인하세요.")
        # SDK/controller protection remains in force. A previous NO_SOLUTION
        # or unreached target is not a PC-side prerequisite for motor enable.
        robot.enable(timeout=0.0)
        status = robot.get_arm_status()
        if recent(status) and status.msg.arm_status == 1:
            raise RuntimeError("활성화 요청 중 EMERGENCY_STOP이 보고됐습니다. 급정지를 자동 해제하지 않습니다.")
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if all(recent(d) and d.timestamp > requested_at
               and d.msg.foc_status.driver_enable_status for d in drivers):
            return
        if time.monotonic() >= deadline:
            pending = [f"J{i}" for i, d in enumerate(drivers, 1)
                       if not (recent(d) and d.timestamp > requested_at
                               and d.msg.foc_status.driver_enable_status)]
            raise RuntimeError("enable을 요청했지만 모터 활성화를 확인하지 못했습니다: " + ", ".join(pending))
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    args = parser.parse_args()
    robot = None
    diagnostics = None
    try:
        print("모터를 활성화합니다. 브레이크 해제 시 자세가 달라질 수 있으므로 팔과 도구 주변을 비우세요.")
        diagnostics = SessionDiagnostics()
        print("실행 진단 로그:", diagnostics.path.resolve())
        diagnostics.event(None, "start", command="enable")
        robot, version, profile = connect_nero(args.channel)
        print(f"펌웨어: {version} / SDK 드라이버: {profile}")
        time.sleep(1.0)
        diagnostics.event(robot, "connected")
        status = robot.get_arm_status()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if recent(status) and status.msg.arm_status == 1:
            raise RuntimeError("EMERGENCY_STOP 상태입니다. enable은 급정지를 해제하지 않습니다. nero_reset.py로 상태를 확인하세요.")
        if (recent(status)
                and all(recent(d) and d.msg.foc_status.driver_enable_status for d in drivers)):
            print("이미 모터 7개가 활성화되어 있습니다.")
        else:
            print("enable 요청을 보냅니다.")
            diagnostics.event(robot, "enable_requested")
            enable_only(robot)
        print_status(robot)
        print("enable 완료: 모터 7개 활성화를 확인했습니다.")
        status = robot.get_arm_status()
        if recent(status) and status.msg.arm_status != 0:
            print(f"모터는 켜졌지만 arm_status={status.msg.arm_status} 표시는 남아 있습니다. 활성화와 오류 해제는 별개입니다.")
            if status.msg.arm_status == 2:
                print("NO_SOLUTION 복구 시도: python nero_recover.py --execute (현재 관절각을 목표로 전송)")
        diagnostics.event(robot, "enabled")
        print("프로그램을 종료해도 모터는 활성화 상태로 남습니다.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("enable 확인 중단:", str(exc) or "사용자 중단")
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=str(exc) or "사용자 중단")
        print("자동 reset/급정지/disable은 보내지 않았습니다. 일부 모터가 켜졌을 수 있으므로 표시 상태를 확인하세요.")
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
