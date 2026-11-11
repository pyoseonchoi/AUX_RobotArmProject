"""Nero connection/status check. Never enables motors or sends motion commands."""

import argparse
import math
import time

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
    resolve_firmware_profile,
)
from nero_feedback_clock import feedback_time


class FeedbackUnavailableError(RuntimeError):
    """Required feedback is absent, stale, or predates the new session check."""


def recent(feedback):
    # Copy the timestamp before reading the clock: the receiver updates the
    # shared feedback object concurrently.
    if feedback is None:
        return False
    timestamp = feedback.timestamp
    return math.isfinite(timestamp) and 0 <= feedback_time() - timestamp < 0.5


def require_feedback(feedbacks, *, newer_than=None):
    """Check every required CAN feedback item against the same age limit."""
    stamps = [(name, None if feedback is None else feedback.timestamp) for name, feedback in feedbacks]
    now = feedback_time()
    missing = []
    for name, timestamp in stamps:
        if timestamp is None:
            missing.append(f"{name}=미수신")
            continue
        if not math.isfinite(timestamp):
            missing.append(f"{name}=유효하지 않은 수신 시각")
            continue
        age = now - timestamp
        if age < 0:
            missing.append(f"{name}=수신 시계 불일치(현재 시각보다 {-age * 1000:.6f}ms 앞섬)")
        elif age >= 0.5:
            missing.append(f"{name}=마지막 수신 후 {age:.6f}초")
        elif newer_than is not None and timestamp <= newer_than:
            missing.append(f"{name}=새 수신 프레임 대기")
    if missing:
        raise FeedbackUnavailableError("필요한 최신 피드백이 없습니다: " + "; ".join(missing))


def connect_nero(channel="0"):
    """Query firmware, then reconnect with its matching Windows CAN driver."""
    robot = AgxArmFactory.create_arm(
        create_agx_arm_config(
            robot=ArmModel.NERO, interface="agx_cando", channel=channel
        )
    )
    try:
        robot.connect()
        firmware = None
        for _ in range(3):
            firmware = robot.get_firmware(timeout=1.0, min_interval=0.0)
            if firmware is not None:
                break
            time.sleep(0.2)
        if firmware is None:
            raise RuntimeError(
                "펌웨어 응답이 없습니다. 전원, USB-CAN, 채널을 확인하세요. "
                "이 프로그램은 응답을 얻기 위해 모터를 활성화하지 않습니다."
            )
        version = firmware["software_version"]
        profile = resolve_firmware_profile(ArmModel.NERO, version)
    finally:
        robot.disconnect()

    robot = AgxArmFactory.create_arm(
        create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=profile,
            interface="agx_cando",
            channel=channel,
        )
    )
    try:
        robot.connect()
    except Exception:
        robot.disconnect()
        raise
    return robot, version, profile


def print_status(robot):
    joint = robot.get_joint_angles()
    status = robot.get_arm_status()
    pose = robot.get_flange_pose()
    if joint is None:
        print("관절 피드백: 아직 수신되지 않았습니다.")
    else:
        values = list(joint.msg)
        print("관절 각도 (deg):", [round(math.degrees(v), 3) for v in values])
        print("관절 각도 (rad):", [round(v, 6) for v in values])
        limits = robot.get_config()["joint_limits"]
        for i, value in enumerate(values, 1):
            low, high = limits["joint{}".format(i)]
            if not low <= value <= high:
                print(
                    "주의: J{} 현재 {:.3f}도, SDK 범위 {:.3f} ~ {:.3f}도. "
                    "이동 전에 확인하세요.".format(
                        i, math.degrees(value), math.degrees(low), math.degrees(high)
                    )
                )
    print("모터 활성화 피드백:", robot.get_joints_enable_status_list())
    if status is None:
        print("상태 피드백: 아직 수신되지 않았습니다.")
    else:
        print(status.msg)
    if pose is not None:
        print("플랜지 위치/자세 (m, rad):", [round(v, 6) for v in pose.msg])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--watch", action="store_true", help="1초 간격으로 계속 표시")
    args = parser.parse_args()
    try:
        robot, version, profile = connect_nero(args.channel)
    except Exception as exc:
        parser.exit(status=1, message="상태 조회 연결 실패: {}\n".format(exc))
    try:
        print("펌웨어: {} / SDK 드라이버: {}".format(version, profile))
        print("상태 조회만 실행합니다. 모터 활성화/이동 명령은 보내지 않습니다.")
        time.sleep(1.0)
        while True:
            print_status(robot)
            if not args.watch:
                break
            print()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("상태 조회를 종료합니다.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
