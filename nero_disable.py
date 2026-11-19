"""Request motor disable regardless of joint range or previous target status."""

import argparse
import math
import time

from nero_diagnostics import SessionDiagnostics
from nero_read_status import connect_nero, feedback_time, print_status, recent


def disable_only(robot, timeout=5.0):
    """Send first; require fresh post-request feedback from every motor to confirm."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("비활성화 대기 시간은 0보다 큰 유한한 값이어야 합니다.")
    deadline = time.monotonic() + timeout
    requested_at = feedback_time()
    while True:
        # No joint-angle, arm-status or target-reached prerequisite. The SDK
        # sends the request; its return value may be based on cached feedback.
        robot.disable()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if all(recent(d) and d.timestamp > requested_at
               and not d.msg.foc_status.driver_enable_status for d in drivers):
            return
        if time.monotonic() >= deadline:
            pending = [f"J{i}" for i, d in enumerate(drivers, 1)
                       if not (recent(d) and d.timestamp > requested_at
                               and not d.msg.foc_status.driver_enable_status)]
            raise RuntimeError("disable을 요청했지만 비활성화를 확인하지 못했습니다: " + ", ".join(pending))
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    args = parser.parse_args()
    robot = None
    diagnostics = None
    try:
        print("모터를 비활성화합니다. 팔/도구의 지지 상태를 확인하세요. 팔이 처질 수 있습니다.")
        diagnostics = SessionDiagnostics()
        print("실행 진단 로그:", diagnostics.path.resolve())
        diagnostics.event(None, "start", command="disable")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        diagnostics.event(robot, "disable_requested")
        disable_only(robot)
        diagnostics.event(robot, "disabled")
        print_status(robot)
        print("모터 7개 비활성화 확인 완료. 다시 움직이려면 enable 단계부터 실행하세요.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("비활성화 확인 중단:", str(exc) or "사용자 중단")
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=str(exc) or "사용자 중단")
        print("현재 활성화 여부를 상태 조회로 다시 확인하세요.")
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
