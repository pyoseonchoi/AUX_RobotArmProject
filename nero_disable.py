"""Disable all Nero motors after checking stopped status; never sends move targets."""

import argparse
import time

from nero_enable import recent
from nero_read_status import connect_nero, print_status


def disable_only(robot):
    status = robot.get_arm_status()
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    if not recent(status) or not all(recent(d) for d in drivers):
        raise RuntimeError("최근 상태/드라이버 피드백이 없습니다. 비활성화하지 않습니다.")
    if status.msg.motion_status != 0:
        raise RuntimeError("목표 도달 상태가 아닙니다. disable을 이동 중 정지용으로 사용하지 마세요.")
    deadline = time.monotonic() + 5.0
    while True:
        robot.disable()
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        if all(recent(d) and not d.msg.foc_status.driver_enable_status for d in drivers):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("7개 모터의 비활성화를 확인하지 못했습니다. 상태를 다시 조회하세요.")
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    args = parser.parse_args()
    robot = None
    try:
        print("모터를 비활성화합니다. 팔/도구의 지지 상태를 확인하세요. 팔이 처질 수 있습니다.")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        disable_only(robot)
        print_status(robot)
        print("모터 7개 비활성화 확인 완료. 다시 움직이려면 enable 단계부터 실행하세요.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("비활성화 확인 중단:", str(exc) or "사용자 중단")
        print("현재 활성화 여부를 상태 조회로 다시 확인하세요.")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
