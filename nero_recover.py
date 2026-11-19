"""Clear a retained NO_SOLUTION flag with a straight-line (move_l) command.

On firmware 1.12 the flag belongs to the Cartesian planner. A joint move
(move_j) leaves it set: the home return of 2026-11-17 18:21 completed with the
flag still on, and the earlier version of this script, which sent the current
joint angles, could not clear it (18:29). The next accepted move_l cleared it
within 0.5 s (2026-11-16 23:50).
"""

import argparse
import math
import time

from nero_diagnostics import SessionDiagnostics, diagnose
from nero_move_joint import ControllerRejected, _motion_snapshot, require_planned_mode
from nero_move_linear import move_linear_target
from nero_read_status import (FeedbackUnavailableError, connect_nero, feedback_time, print_status,
                              read_flange_pose)


NUDGE_MM = 1.0  # Size of the fallback straight move (down, then back).


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


def wait_for_normal(robot, *, newer_than, timeout=1.0):
    """True once fresh status feedback (newer than ``newer_than``) reports NORMAL.

    False if fresh feedback keeps reporting the flag; raises if fresh feedback
    never arrives, so a lost stream is not mistaken for a flag that stayed.
    """
    deadline = time.monotonic() + timeout
    seen, missing = False, None
    while True:
        try:
            _, state, stamps = recovery_sample(robot)
            if all(stamp > newer_than for stamp in stamps):
                seen = True
                if state[1] == 0:
                    return True
        except FeedbackUnavailableError as exc:
            missing = exc
        if time.monotonic() >= deadline:
            if not seen and missing is not None:
                raise missing
            return False
        time.sleep(0.02)


def recover_no_solution(robot, timeout=5.0):
    """Send straight-line targets until the flag clears; no reset/enable/disable.

    Attempt 1 sends the measured flange pose itself (no motion). Attempt 2
    moves the flange NUDGE_MM down and back up (up instead, if the flange is
    low). Both go through move_linear_target, so feedback, fault and arrival
    checks and the emergency-stop policy are the same as for any move_l.
    Returns the joint angles once fresh feedback shows NORMAL.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("복구 확인 시간은 0보다 큰 유한한 초 값이어야 합니다.")
    angles, state, _ = recovery_sample(robot)
    if state[1] == 0:
        print("이미 NORMAL입니다. 복구 명령을 보내지 않았습니다.")
        return angles
    if not robot.get_auto_set_motion_mode_enabled():
        raise RuntimeError("SDK 자동 이동 모드 전환이 꺼져 있어 목표를 보내지 않습니다.")
    pose, _ = read_flange_pose(robot)
    nudged = list(pose)
    nudged[2] += (NUDGE_MM if pose[2] < 0.06 else -NUDGE_MM) / 1000.0
    attempts = (("현재 위치를 그대로 직선 목표로 보냅니다 (움직임 없음).", [pose]),
                ("플랜지를 {:g}mm 내렸다가 되돌립니다.".format(NUDGE_MM), [nudged, pose]))
    for number, (description, targets) in enumerate(attempts, 1):
        print("복구 시도 {}: {} 속도 5%.".format(number, description))
        diagnose(robot, "recovery_attempt", attempt=number, targets_m_rad=targets)
        try:
            for index, target in enumerate(targets):
                move_linear_target(robot, target, timeout=timeout, speed_percent=5, settled=index > 0)
        except ControllerRejected as exc:
            # The controller kept the flag and did not move: try the next way.
            print("이 방법으로는 표시가 남았습니다:", exc)
            continue
        if wait_for_normal(robot, newer_than=feedback_time()):
            angles, _, _ = recovery_sample(robot)
            diagnose(robot, "recovery_confirmed", attempt=number)
            print("복구 확인: NORMAL / 모터 7개 활성. 자세는 그대로입니다.")
            return angles
        print("직선 명령은 끝났지만 NO_SOLUTION 표시가 남았습니다.")
    raise RuntimeError("직선 명령으로도 NO_SOLUTION 표시가 지워지지 않았습니다. "
                       "이 표시는 이동을 막지 않습니다(nero_goto, nero_home은 그대로 실행 가능). "
                       "표시를 반드시 지워야 하면 팔을 받친 상태에서 전원을 껐다 켜세요.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="0")
    parser.add_argument("--execute", action="store_true",
                        help="직선 명령으로 복구 시도 (현재 위치 그대로, 안 되면 1mm 내렸다 되돌림)")
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
            print("팔을 멈추고 주변을 비우세요. 필요하면 플랜지가 {:g}mm 움직였다 돌아옵니다.".format(NUDGE_MM))
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
