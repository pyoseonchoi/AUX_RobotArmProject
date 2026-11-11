"""Move to absolute base-frame XYZ in millimeters, keeping flange orientation."""

import argparse
import math
import time

from nero_move_linear import FeedbackUnavailableError, flange_pose, move_linear_target
from nero_read_status import connect_nero, feedback_time


def make_target(start, xyz_mm, max_distance_mm=20.0):
    """Omitted coordinates retain the displayed position; angles stay unchanged."""
    if len(start) != 6 or not all(math.isfinite(value) for value in start):
        raise ValueError("현재 위치/방향 6개 값이 필요합니다.")
    if len(xyz_mm) != 3:
        raise ValueError("목표 좌표는 X, Y, Z 세 값입니다.")
    if not math.isfinite(max_distance_mm) or max_distance_mm <= 0:
        raise ValueError("최대 이동 거리는 0보다 큰 유한한 mm 값이어야 합니다.")
    target = list(start)
    for i, value in enumerate(xyz_mm):
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("좌표에는 유한한 숫자를 입력하세요.")
            target[i] = value / 1000.0
    distance_mm = math.sqrt(sum((a - b) ** 2 for a, b in zip(start[:3], target[:3]))) * 1000
    if distance_mm > max_distance_mm + 1e-9:
        raise ValueError(
            f"이동 거리 {distance_mm:.3f}mm가 현재 한도 {max_distance_mm:g}mm보다 큽니다. "
            "단위와 원점을 확인하세요. 큰 이동은 --max-distance-mm로 한도를 명시할 수 있습니다."
        )
    return target, distance_mm


def read_xyz(start):
    print("목표 절대좌표를 mm로 입력하세요. Enter만 누르면 표시된 좌표를 유지합니다.")
    print("입력 완료 후 실제 직선 이동을 실행합니다.")
    values = []
    for axis, value in zip(("X", "Y", "Z"), start[:3]):
        answer = input(f"목표 {axis} (현재 {value * 1000:.3f}mm): ").strip()
        values.append(float(answer) if answer else None)
    return values


def wait_for_flange_pose(robot, *, newer_than=None, timeout=2.0):
    """Wait for fresh feedback before moving; never extend its maximum age."""
    deadline = time.monotonic() + timeout
    last_reason = "새 플랜지 피드백 대기 중"
    while True:
        if not robot.is_connected():
            raise RuntimeError("CAN 연결이 닫혀 있어 최신 좌표를 조회할 수 없습니다.")
        error = robot.get_comm_error()
        if error is not None:
            raise RuntimeError(f"CAN 통신 오류로 좌표 조회를 중단합니다: {error}")
        try:
            pose, timestamp = flange_pose(robot, newer_than=newer_than)
            if newer_than is None or timestamp > newer_than:
                return pose, timestamp
            last_reason = f"새 수신 시각이 필요합니다. 마지막 수신 후 {feedback_time() - timestamp:.6f}초"
        except FeedbackUnavailableError as exc:
            last_reason = str(exc)
        if time.monotonic() >= deadline:
            reader = getattr(robot.get_context(), "_read_th", None)
            reader_alive = reader is not None and reader.is_alive()
            raise FeedbackUnavailableError(
                f"{timeout:g}초 안에 최신 플랜지 피드백을 받지 못했습니다. {last_reason}. "
                f"CAN 수신 스레드 실행 여부={reader_alive}. "
                "전원/USB-CAN 연결과 다른 프로그램의 동일 장치 사용 여부를 확인하세요."
            )
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--x", type=float, help="베이스 기준 절대 X좌표(mm)")
    parser.add_argument("--y", type=float, help="베이스 기준 절대 Y좌표(mm)")
    parser.add_argument("--z", type=float, help="베이스 기준 절대 Z좌표(mm)")
    parser.add_argument("--preview", action="store_true", help="좌표 조회/목표 계산만 하고 이동 명령은 보내지 않음")
    parser.add_argument("--max-distance-mm", type=float, default=20.0,
                        help="실수 방지용 직선 이동 거리 한도(mm), 기본 20. 도달 가능성 검사는 아님")
    args = parser.parse_args()
    robot = None
    try:
        if not math.isfinite(args.max_distance_mm) or args.max_distance_mm <= 0:
            raise ValueError("최대 이동 거리는 0보다 큰 유한한 mm 값이어야 합니다.")
        if not all(value is None or math.isfinite(value) for value in (args.x, args.y, args.z)):
            raise ValueError("목표 좌표는 유한한 숫자여야 합니다.")
        print("기준: 로봇 베이스 / 위치 단위: mm / 제어점: 플랜지(도구 장착면)")
        if args.preview:
            print("미리보기: 상태 조회와 목표 계산만 실행합니다.")
        else:
            print("실제 직선 이동을 실행합니다. 팔과 도구가 움직일 공간을 비우세요.")
        robot, _, _ = connect_nero(args.channel)
        time.sleep(1.0)
        start, _ = wait_for_flange_pose(robot)
        print("현재 XYZ(mm):", [round(value * 1000, 3) for value in start[:3]])
        print("현재 방향(deg):", [round(math.degrees(value), 3) for value in start[3:]])
        coordinates = [args.x, args.y, args.z]
        if all(value is None for value in coordinates):
            if args.preview:
                return 0
            # Console entry may take arbitrarily long. Do not retain a live
            # receive session or trust its cached pose across user input.
            robot.disconnect()
            robot = None
            print("상태 조회 연결을 닫고 좌표를 입력받습니다. 모터를 비활성화하지는 않습니다.")
            coordinates = read_xyz(start)
        target, distance_mm = make_target(start, coordinates, args.max_distance_mm)
        print("목표 XYZ(mm):", [round(value * 1000, 3) for value in target[:3]])
        print(f"입력 시작 위치와 목표 사이 직선 거리: {distance_mm:.3f}mm")
        if args.preview:
            print("미리보기 완료. 이동 명령은 보내지 않았습니다. 도달 가능성/충돌 검사는 수행하지 않습니다.")
            return 0
        if robot is None:
            print("입력이 끝났습니다. 다시 연결해 최신 위치/상태를 확인합니다.")
            robot, _, _ = connect_nero(args.channel)
            time.sleep(1.0)
        # Require a newly received frame, not just the pre-input snapshot.
        # Preserve the orientation measured just before moving, and recheck
        # the distance limit from the newly received position.
        current, _ = wait_for_flange_pose(robot, newer_than=feedback_time())
        target, distance_mm = make_target(current, [value * 1000 for value in target[:3]], args.max_distance_mm)
        if distance_mm < 0.5:
            print("현재 위치가 목표에서 0.5mm 미만입니다. 이동 명령을 보내지 않습니다.")
            return 0
        move_linear_target(robot, target)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("좌표 이동 중단:", str(exc) or "사용자 중단")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
