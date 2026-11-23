"""Save the measured home posture, preview it, or return to it with --execute."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

from nero_diagnostics import SessionDiagnostics
from nero_kinematics import height_floor_mm, height_limit_text, joint_move_trace
from nero_move_joint import move_joint_target
from nero_move_linear import orientation_error
from nero_read_status import (FeedbackUnavailableError, connect_nero, feedback_time,
                              read_flange_pose, require_feedback)


DEFAULT_POSITION = Path(__file__).resolve().with_name("nero_home_position.json")
JOINT_PARTS = ("joint_12", "joint_34", "joint_56", "joint_7")


def read_posture(robot, *, newer_than=None):
    """Read-only; capturing a home does not require enabled motors."""
    if not robot.is_connected() or robot.get_comm_error() is not None:
        raise RuntimeError("CAN 연결/통신 상태를 확인하세요.")
    joint = robot.get_joint_angles()
    required = [("관절 통합값", joint)]
    parser = getattr(robot, "_parser", None)
    if parser is not None:
        required.extend((name, getattr(parser, name, None)) for name in JOINT_PARTS)
    require_feedback(required, newer_than=newer_than)
    angles = list(joint.msg)
    if len(angles) != 7 or not all(math.isfinite(q) for q in angles):
        raise RuntimeError("유효한 관절각 7개가 필요합니다.")
    for i, q in enumerate(angles, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= q <= high:
            raise RuntimeError(f"J{i} 각도가 SDK 범위 밖입니다.")
    pose, _ = read_flange_pose(robot, newer_than=newer_than)
    return angles, pose


def capture_home(robot, version, profile, timeout=3.0):
    """Require renewed joint/pose packets and a stationary half-second window."""
    deadline = time.monotonic() + timeout
    marker = feedback_time()
    started = None
    lows = highs = first_pose = None
    samples = 0
    while time.monotonic() < deadline:
        try:
            angles, pose = read_posture(robot, newer_than=marker)
        except FeedbackUnavailableError:
            time.sleep(0.02)
            continue
        marker = feedback_time()
        if started is None:
            started = time.monotonic()
            lows, highs, first_pose = angles.copy(), angles.copy(), pose
        lows = [min(a, b) for a, b in zip(lows, angles)]
        highs = [max(a, b) for a, b in zip(highs, angles)]
        if (max(h - l for l, h in zip(lows, highs)) > math.radians(0.05)
                or math.dist(first_pose[:3], pose[:3]) > 0.0005
                or orientation_error(first_pose, pose) > math.radians(0.1)):
            raise RuntimeError("자세가 변하고 있어 초기화 위치를 저장하지 않습니다. 팔을 멈춘 뒤 다시 조회하세요.")
        samples += 1
        if samples >= 3 and time.monotonic() - started >= 0.5:
            return dict(schema_version=1, coordinate_frame="robot_base", control_point="flange",
                        captured_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                        firmware=version, sdk_profile=str(profile),
                        joints_rad=angles, flange_pose_m_rad=pose)
        time.sleep(0.02)
    raise FeedbackUnavailableError("새 관절/플랜지 피드백으로 정지 자세를 확인하지 못했습니다.")


def load_home(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if (data.get("schema_version") != 1 or data.get("coordinate_frame") != "robot_base"
            or data.get("control_point") != "flange"):
        raise ValueError("초기화 위치 파일의 형식/좌표계가 올바르지 않습니다.")
    for key, size in (("joints_rad", 7), ("flange_pose_m_rad", 6)):
        values = data.get(key)
        if (not isinstance(values, list) or len(values) != size
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in values)):
            raise ValueError(f"초기화 위치의 {key} 값이 올바르지 않습니다.")
    return data


def show_home(home):
    pose = home["flange_pose_m_rad"]
    print("초기화 XYZ (mm, 베이스 기준 / 플랜지): X={:.3f}, Y={:.3f}, Z={:.3f}".format(
        *[v * 1000 for v in pose[:3]]))
    print("초기화 RPY (deg):", [round(math.degrees(v), 3) for v in pose[3:]])
    print("초기화 관절각 (deg):", [round(math.degrees(v), 3) for v in home["joints_rad"]])


DEFAULT_SPEED_PERCENT = 7  # Same default and upper bound as nero_goto.py --speed.
MAX_SPEED_PERCENT = 20
START_DELAY_S = 3.0        # Time to abort with Ctrl+C before a real return starts.


def check_return_path(angles, home, min_z_mm):
    """Flange path length (mm) and lowest elbow/wrist/flange height (mm) of the return.

    Replaces a fixed joint-travel cap: a large swing is allowed as long as the
    arm does not dip toward the table on the way. An arm that already starts
    below min_z_mm may return as long as it never goes lower than it is now.
    No collision check otherwise.
    """
    path_mm, lowest_mm = joint_move_trace(angles, home["joints_rad"], together=True)
    floor_mm = height_floor_mm(angles, min_z_mm)
    if lowest_mm < floor_mm:
        raise RuntimeError("복귀 중 팔꿈치·손목·플랜지가 높이 {:.0f}mm까지 내려갑니다 ({}). "
                           "팔을 먼저 위쪽으로 옮긴 뒤 복귀하세요.".format(
                               lowest_mm, height_limit_text(floor_mm, min_z_mm)))
    return path_mm, lowest_mm


def return_home(robot, home, *, timeout=60.0, max_joint_move_deg=360.0, speed_percent=5, min_z_mm=50.0):
    target = home["joints_rad"]
    angles, pose = read_posture(robot)
    if (max(abs(a - b) for a, b in zip(angles, target)) < math.radians(0.1)
            and math.dist(pose[:3], home["flange_pose_m_rad"][:3]) < 0.002
            and orientation_error(pose, home["flange_pose_m_rad"]) < math.radians(1)):
        print("이미 초기화 위치·자세입니다. 이동 명령을 보내지 않습니다.")
        return
    check_return_path(angles, home, min_z_mm)
    move_joint_target(robot, target, timeout=timeout, max_step_degrees=max_joint_move_deg,
                      speed_percent=speed_percent, allow_long_move=True, lead_degrees=2.0)
    _, pose = read_posture(robot)
    error_mm = math.dist(pose[:3], home["flange_pose_m_rad"][:3]) * 1000
    if error_mm >= 2 or orientation_error(pose, home["flange_pose_m_rad"]) >= math.radians(1):
        raise RuntimeError(f"관절 목표에는 도달했지만 저장한 플랜지 자세와 다릅니다 (위치 오차 {error_mm:.3f}mm).")
    print(f"초기화 위치 복귀 완료. XYZ 오차 {error_mm:.3f}mm.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--position", type=Path, default=DEFAULT_POSITION, help="초기화 위치 JSON 파일")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--save-current", action="store_true", help="현재 자세를 새 파일로 저장 (이동 없음, 덮어쓰기 없음)")
    action.add_argument("--execute", action="store_true", help="저장한 초기화 자세로 실제 이동")
    parser.add_argument("--timeout", type=float, default=60.0, help="목표에 가까워지지 않는 최대 시간(초)")
    parser.add_argument("--max-joint-move-deg", type=float, default=360.0,
                        help="관절별 최대 이동각(도), 0~360. 기본 360(사실상 제한 없음)")
    parser.add_argument("--min-z-mm", type=float, default=50.0,
                        help="복귀 중 팔꿈치·손목·플랜지의 최저 허용 높이(mm). 기본 50. 충돌 검사는 아님")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED_PERCENT,
                        help="속도 비율(%%), 1~{}. 기본 {}".format(MAX_SPEED_PERCENT, DEFAULT_SPEED_PERCENT))
    args = parser.parse_args()
    robot = diagnostics = None
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout은 0보다 큰 유한한 값이어야 합니다.")
        if not 1 <= args.speed <= MAX_SPEED_PERCENT:
            raise ValueError("속도 비율은 1~{}%입니다.".format(MAX_SPEED_PERCENT))
        if not math.isfinite(args.max_joint_move_deg) or not 0 < args.max_joint_move_deg <= 360:
            raise ValueError("관절 이동 한도는 0보다 크고 360도 이하여야 합니다.")
        if not math.isfinite(args.min_z_mm):
            raise ValueError("최저 높이는 유한한 mm 값이어야 합니다.")
        if args.save_current:
            if args.position.exists():
                raise FileExistsError("이미 저장한 초기화 위치입니다. 새 위치는 --position으로 다른 파일을 지정하세요.")
            home = None
        else:
            home = load_home(args.position)
            show_home(home)
        diagnostics = SessionDiagnostics()
        robot, version, profile = connect_nero(args.channel)
        robot._nero_diagnostics = diagnostics
        time.sleep(1.0)
        diagnostics.event(robot, "start", command="home", options={**vars(args), "position": str(args.position)})
        if args.save_current:
            home = capture_home(robot, version, profile)
            with args.position.open("x", encoding="utf-8") as stream:
                json.dump(home, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            show_home(home)
            print("초기화 위치 저장 완료:", args.position.resolve())
        else:
            angles, pose = read_posture(robot)
            print("현재 XYZ (mm):", [round(v * 1000, 3) for v in pose[:3]])
            travel = max(abs(a - b) for a, b in zip(angles, home["joints_rad"]))
            print("최대 관절 이동각: {:.3f}도 / 한도 {:g}도".format(math.degrees(travel), args.max_joint_move_deg))
            print("복귀 경로: 관절 동시 이동, 속도 {}%. 플랜지의 경로는 직선이 아니며 장애물 회피는 하지 않습니다.".format(
                args.speed))
            path_mm, lowest_mm = joint_move_trace(angles, home["joints_rad"], together=True)
            print("복귀 중 플랜지 이동 거리 약 {:.0f}mm / 팔꿈치·손목·플랜지 최저 높이 {:.0f}mm ({})".format(
                path_mm, lowest_mm, height_limit_text(height_floor_mm(angles, args.min_z_mm), args.min_z_mm)))
            if math.degrees(travel) > 90:
                print("큰 관절 이동입니다. 팔 전체가 넓게 휘돌아 가므로 주변 공간을 확인하세요.")
            if args.execute:
                check_return_path(angles, home, args.min_z_mm)
                if travel >= math.radians(0.1):
                    print("이미 활성화된 모터로 복귀합니다. 자동 enable/reset은 하지 않습니다.")
                    print("{:g}초 후 복귀를 시작합니다. 중단하려면 Ctrl+C.".format(START_DELAY_S))
                    time.sleep(START_DELAY_S)
                return_home(robot, home, timeout=args.timeout, max_joint_move_deg=args.max_joint_move_deg,
                            speed_percent=args.speed, min_z_mm=args.min_z_mm)
            else:
                print("미리보기 완료. 실제 복귀: python nero_home.py --execute")
        diagnostics.event(robot, "completed")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("초기화 위치 처리 중단:", str(exc) or "사용자 중단")
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=str(exc) or "사용자 중단")
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
