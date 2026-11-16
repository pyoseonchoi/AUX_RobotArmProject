"""Try replacing a retained NO_SOLUTION target with the measured joint posture."""

import argparse
import math
import time

from nero_diagnostics import SessionDiagnostics, diagnose
from nero_move_joint import (
    RejectionWatch, _motion_snapshot, check_before_move, require_planned_mode,
    stop_failed_move,
)
from nero_read_status import connect_nero, feedback_time, print_status


def recovery_sample(robot):
    """Fresh, enabled, ordinary CAN feedback; only NORMAL/NO_SOLUTION allowed."""
    if not robot.is_connected() or robot.get_comm_error() is not None:
        raise RuntimeError("CAN 연결/통신 오류로 복구를 중단합니다.")
    angles, _, _ = _motion_snapshot(robot)
    status = robot.get_arm_status()
    state = (int(status.msg.ctrl_mode), int(status.msg.arm_status),
             int(status.msg.mode_feedback), int(status.msg.motion_status))
    if state[1] not in (0, 2):
        raise RuntimeError("이 복구는 NO_SOLUTION에만 사용합니다: {}".format(status.msg.arm_status))
    require_planned_mode(robot)
    for i, angle in enumerate(angles, 1):
        low, high = robot.get_config()["joint_limits"][f"joint{i}"]
        if not low <= angle <= high:
            raise RuntimeError(f"J{i} 현재 각도가 SDK 범위 밖입니다.")
        errors = status.msg.err_status
        if (getattr(errors, f"joint_{i}_angle_limit")
                or getattr(errors, f"communication_status_joint_{i}")):
            raise RuntimeError(f"J{i} 제한/통신 오류가 보고됐습니다.")
    stamps = tuple(getattr(robot._parser, name).timestamp for name in
                   ("joint_12", "joint_34", "joint_56", "joint_7"))
    stamps += (status.timestamp,)
    stamps += tuple(robot.get_driver_states(i).timestamp for i in range(1, 8))
    return angles, state, stamps


def recover_no_solution(robot, timeout=3.0):
    """One joint target, no reset/enable/disable. Firmware clearance is not assumed."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("복구 확인 시간은 0보다 큰 유한한 초 값이어야 합니다.")
    target, state, _ = recovery_sample(robot)
    if state[1] == 0:
        print("이미 NORMAL입니다. 복구 명령을 보내지 않았습니다.")
        return target
    check_before_move(robot)
    target, state, _ = recovery_sample(robot)
    if state[1] == 0:
        print("NORMAL로 바뀌었습니다. 복구 명령을 보내지 않았습니다.")
        return target
    if not robot.get_auto_set_motion_mode_enabled():
        raise RuntimeError("SDK 자동 이동 모드 전환이 꺼져 있어 목표를 보내지 않습니다.")
    print("현재 관절각(deg):", [round(math.degrees(q), 3) for q in target])
    print("이 각도를 그대로 MOVE_J 목표로 한 번 보냅니다. 속도 비율 5%입니다.")
    watch = RejectionWatch(robot, target)
    started = False
    try:
        started = True
        robot.set_speed_percent(5)
        robot.move_j(list(target))
        # Require telemetry newer than completion of all four target packets.
        sent_at = feedback_time()
        watch.target_sent()
        diagnose(robot, "recovery_target_sent", target_rad=target)
        deadline = time.monotonic() + timeout
        previous = None
        stable_since = None
        samples = 0
        while True:
            angles, state, stamps = recovery_sample(robot)
            error = max(abs(q - t) for q, t in zip(angles, target))
            if error > math.radians(0.5):
                raise RuntimeError("복구 중 관절이 기준보다 0.5도 넘게 변했습니다.")
            watch.check(angles)
            confirmed = state == (1, 0, 1, 0) and error < math.radians(0.1)
            if not confirmed:
                samples, stable_since = 0, None
            if (all(stamp > sent_at for stamp in stamps)
                    and (previous is None or all(a > b for a, b in zip(stamps, previous)))):
                previous = stamps
                if confirmed:
                    samples += 1
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if samples >= 3 and time.monotonic() - stable_since >= 0.2:
                        diagnose(robot, "recovery_confirmed", target_rad=target)
                        print("복구 확인: NORMAL / MOVE_J / 목표 도달 / 모터 7개 활성.")
                        return angles
            if time.monotonic() >= deadline:
                raise RuntimeError("복구 미확인: 새 NORMAL / MOVE_J / 목표 도달 피드백이 없습니다.")
            time.sleep(0.02)
    except BaseException as exc:
        if started:
            stop_failed_move(robot, exc, watch)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--execute", action="store_true", help="현재 관절각을 목표로 보내 복구 시도")
    args = parser.parse_args()
    robot = diagnostics = None
    try:
        diagnostics = SessionDiagnostics()
        print("실행 진단 로그:", diagnostics.path.resolve())
        diagnostics.event(None, "start", command="recover_no_solution", execute=args.execute)
        robot, version, profile = connect_nero(args.channel)
        robot._nero_diagnostics = diagnostics
        print(f"펌웨어: {version} / SDK: {profile}")
        time.sleep(1.0)
        diagnostics.event(robot, "connected")
        if args.execute:
            print("팔을 멈추고 주변을 비우세요. 현재 자세 유지 목표라도 미세한 보정 움직임이 생길 수 있습니다.")
            recover_no_solution(robot)
        else:
            print("조회만 합니다. 복구 시도: python nero_recover.py --execute")
        print_status(robot)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) or "사용자 중단"
        print("NO_SOLUTION 복구 중단:", reason)
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=reason)
        print("해제를 확인하지 못했습니다. 자동 재시도/enable/disable/reset은 하지 않습니다.")
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
