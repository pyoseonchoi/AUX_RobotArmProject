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


def connect_nero(channel="0", *, progress=None):
    """Query firmware, then reconnect with its matching Windows CAN driver."""
    report = progress if progress is not None else lambda message: None
    report(f"USB-CAN 채널 {channel} 열기 / SDK 생성 중...")
    robot = AgxArmFactory.create_arm(
        create_agx_arm_config(
            robot=ArmModel.NERO, interface="agx_cando", channel=channel
        )
    )
    try:
        report("CAN 수신 연결 시작 중...")
        robot.connect()
        firmware = None
        for attempt in range(3):
            report(f"로봇 펌웨어 응답 대기 ({attempt + 1}/3)...")
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
        report("펌웨어 조회 연결 정리 중...")
        robot.disconnect()

    report(f"펌웨어 {version} 확인. {profile} 드라이버로 USB-CAN 다시 여는 중...")
    robot = AgxArmFactory.create_arm(
        create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=profile,
            interface="agx_cando",
            channel=channel,
        )
    )
    try:
        report("버전에 맞는 CAN 수신 연결 시작 중...")
        robot.connect()
    except Exception:
        robot.disconnect()
        raise
    return robot, version, profile


def read_flange_pose(robot, *, newer_than=None):
    """Read all six flange coordinates only when every CAN component is fresh."""
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


def print_status(robot):
    joint = robot.get_joint_angles()
    status = robot.get_arm_status()
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
    try:
        pose, _ = read_flange_pose(robot)
    except RuntimeError as exc:
        print("현재 XYZ 확인 불가:", exc)
    else:
        print("현재 XYZ (mm, 베이스 기준 / 플랜지): X={:.3f}, Y={:.3f}, Z={:.3f}".format(
            *[v * 1000 for v in pose[:3]]))
        print("현재 방향 RPY (deg):", [round(math.degrees(v), 3) for v in pose[3:]])
        print("플랜지 위치/자세 (m, rad):", [round(v, 6) for v in pose])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--watch", action="store_true", help="1초 간격으로 계속 표시")
    args = parser.parse_args()
    print("상태 조회 시작: USB-CAN 연결과 펌웨어를 확인합니다. 모터/이동 명령은 보내지 않습니다.", flush=True)
    try:
        robot, version, profile = connect_nero(args.channel, progress=lambda message: print(message, flush=True))
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
