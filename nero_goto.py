"""Move the flange to an absolute base-frame XYZ with all joints moving together.

Default is a preview: the plan is computed and printed, nothing moves. With
--execute the arm first makes one joint move onto the straight line if it
cannot start there (for example from the upright, singular pose), then drives
the flange along the line with move_l.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import time
import uuid

import numpy as np

from nero_diagnostics import SessionDiagnostics, diagnose
from nero_kinematics import (TOOL_AXES, PlanSettings, forward, plan_joint_move, plan_line,
                             rotation_angle, rpy_to_matrix, segment_distance_mm)
from nero_move_joint import check_before_move, move_joint_target
from nero_move_linear import flange_pose, move_linear_target
from nero_move_to import wait_for_flange_pose
from nero_read_status import FeedbackUnavailableError, connect_nero, require_feedback
from nero_repeat_air import JOINT_PARTS, RunRecorder


START_DELAY_S = 3.0          # Time to abort with Ctrl+C after the plan is shown.
DEFAULT_SPEED_PERCENT = 7    # Default --speed. The speed model was measured at 5% and is scaled.
MAX_SPEED_PERCENT = 20       # Upper bound of --speed.
DEVIATION_WARNING_MM = 20.0  # Drift from the line that is reported but does not stop the arm.
MODEL_TOLERANCE_MM = 3.0     # Allowed gap between the SDK model and flange feedback.
MODEL_TOLERANCE_DEG = 2.0
ENTRY_TOLERANCE_MM = 5.0     # Allowed gap between planned and measured line start.


def read_joints(robot, timeout=2.0):
    """Fresh joint angles for a preview; works while the motors are disabled."""
    deadline = time.monotonic() + timeout
    while True:
        feedback = robot.get_joint_angles()
        parser = getattr(robot, "_parser", None)
        required = [("관절 통합값", feedback)]
        if parser is not None:
            required += [(name, getattr(parser, name, None)) for name in JOINT_PARTS]
        try:
            require_feedback(required)
            angles = list(feedback.msg)
            if len(angles) != 7 or not all(math.isfinite(a) for a in angles):
                raise RuntimeError("7개 관절 각도를 확인하지 못했습니다.")
            return angles
        except FeedbackUnavailableError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(0.02)


def model_gap(joints, pose):
    """How far the SDK kinematic model is from the measured flange pose (mm, deg)."""
    position, rotation = forward(joints)
    return (float(np.linalg.norm(position - np.array(pose[:3]))) * 1000.0,
            math.degrees(rotation_angle(rotation, rpy_to_matrix(*pose[3:]))))


def validate_options(args):
    numbers = (args.timeout, args.max_deviation_mm, args.max_joint_move_deg, args.min_z_mm, args.min_sigma,
               args.joint_update_hz)
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError("옵션에는 유한한 숫자를 입력하세요.")
    if not 10 <= args.joint_update_hz <= 200:
        raise ValueError("관절 목표 갱신 빈도는 10~200Hz입니다.")
    if not all(v is None or math.isfinite(v) for v in (args.x, args.y, args.z)):
        raise ValueError("목표 좌표는 유한한 숫자여야 합니다.")
    if not 1 <= args.speed <= MAX_SPEED_PERCENT:
        raise ValueError("속도 비율은 1~{}%입니다.".format(MAX_SPEED_PERCENT))
    if args.timeout <= 0:
        raise ValueError("timeout은 0보다 커야 합니다.")
    if not 5 <= args.max_deviation_mm <= 100:
        raise ValueError("직선 이탈 정지 한도는 5~100mm입니다.")
    if not 0 < args.max_joint_move_deg <= 360:
        raise ValueError("관절 이동 한도는 0보다 크고 360도 이하여야 합니다.")
    if args.rpy is not None:
        if args.orient is not None:
            raise ValueError("--rpy와 --orient는 함께 쓸 수 없습니다.")
        if (not all(math.isfinite(v) for v in args.rpy) or abs(args.rpy[0]) > 180
                or abs(args.rpy[1]) > 90 or abs(args.rpy[2]) > 180):
            raise ValueError("--rpy는 roll/yaw ±180도, pitch ±90도 범위입니다.")
    if all(v is None for v in (args.x, args.y, args.z)) and args.rpy is None and args.orient is None:
        raise ValueError("목표 좌표(--x/--y/--z) 또는 목표 방향을 하나 이상 지정하세요.")
    settings = PlanSettings(min_z_mm=args.min_z_mm, min_sigma=args.min_sigma,
                            rotate_on_line=args.rotate_on_line, speed_percent=args.speed,
                            joints_together=args.joint_move == "together")
    settings.validate()
    return settings


def orientation_request(args):
    """Keyword arguments for the planner, from --rpy / --orient."""
    if args.rpy is not None:
        return dict(rotation=rpy_to_matrix(*[math.radians(v) for v in args.rpy]))
    if args.orient == "keep":
        return dict(keep_orientation=True)
    if args.orient == "down":
        return dict(aim=(TOOL_AXES[args.tool_axis], (0.0, 0.0, -1.0)))
    return {}


def _mm(values):
    return [round(v * 1000.0, 1) for v in values]


def _deg(values):
    return [round(math.degrees(v), 1) for v in values]


def describe(plan, settings):
    print("--- 이동 계획 (SDK 운동학 모델 기준) ---")
    print("시작 XYZ(mm):", _mm(plan.start_pose[:3]), "/ 방향(deg):", _deg(plan.start_pose[3:]))
    print("목표 XYZ(mm):", _mm(plan.target_pose[:3]), "/ 방향(deg):", _deg(plan.target_pose[3:]))
    print("시작→목표 직선 거리 {:.1f}mm / 방향 변화 {:.1f}도".format(plan.total_mm, plan.orientation_change_deg))
    if not settings.rotate_on_line:
        print("방향은 관절 동시 이동에서 맞추고, 직선 구간에서는 고정합니다.")
    has_line = len(plan.line_joints) > 1
    if plan.needs_joint_move:
        print("[관절 동시 이동 move_j] 가장 많이 도는 관절 {:.1f}도 / 플랜지는 곡선으로 약 {:.1f}mm 이동".format(
            plan.joint_move_deg, plan.joint_move_path_mm))
        if plan.joint_move_deg > 90:
            print("  큰 관절 이동입니다. 팔 전체가 넓게 휘돌아 가므로 주변 공간을 확인하세요.")
        print("  모든 관절이 같은 비율로 움직여 함께 도착합니다." if settings.joints_together
              else "  목표를 한 번에 보냅니다. 이동량이 작은 관절이 먼저 도착합니다.")
        print("  도착 관절각(deg):", _deg(plan.entry_joints))
        print("  도착 XYZ(mm):", _mm(plan.entry_pose[:3]), "/ 방향(deg):", _deg(plan.entry_pose[3:]))
    if has_line:
        share = 100.0 * plan.straight_mm / plan.total_mm if plan.total_mm > 0.5 else 100.0
        print("[직선 이동 move_l] {:.1f}mm (전체 거리의 {:.0f}%)".format(plan.straight_mm, share))
        print("  예상 도착 관절각(deg):", _deg(plan.line_joints[-1]))
        print("  특이점 여유 최소 {:.3f} (한도 {:g}) / 검사점 사이 최대 관절 변화 {:.2f}도".format(
            plan.min_sigma, settings.min_sigma, plan.max_joint_step_deg))
        print("  컨트롤러가 유지할 팔꿈치 회전각 {:.1f}도 / 어깨-손목 축 기울기 최소 {:.1f}도 (한도 {:g}도)".format(
            plan.arm_angle_deg, plan.min_tilt_deg, settings.min_tilt_deg))
        print("  직선 중 필요한 최대 관절 속도 {:.1f}도/초 (한도 {:g}도/초, 속도 {}% 기준 추정)".format(
            plan.peak_joint_speed_deg_s, settings.joint_speed_limit, settings.speed_percent))
    if plan.total_mm > 50 and plan.straight_mm < 0.5 * plan.total_mm:
        print("참고: 직선 구간이 전체의 절반이 안 됩니다. 직선이 베이스 축 가까이를 지나거나 자세 한계에 걸리면,")
        print("      관절이 컨트롤러의 플랜지 속도를 따라가지 못해 직선을 짧게 잡습니다. 직선이 꼭 필요하면 중간 지점을 나눠 지정하세요.")
    print("팔꿈치·손목·플랜지 최저 높이 {:.1f}mm (한도 {:g}mm)".format(plan.min_z_mm, settings.min_z_mm))
    print("주의: 컨트롤러가 고르는 실제 관절 자세는 위 예상과 다를 수 있고, 충돌 검사는 하지 않습니다.")


class LineMonitor:
    """Motion-loop hook: how far the measured flange strays from the straight segment.

    Drift up to ``limit_mm`` only produces a warning (once, past ``warn_mm``):
    an emergency stop is more disruptive than a moderate drift that the
    controller usually corrects on its own (2026-11-17 00:06: 10.8 mm, recovered).
    Past ``limit_mm`` the move fails and the caller requests the stop.
    """

    def __init__(self, robot, start, end, limit_mm, also=None, *, warn_mm=DEVIATION_WARNING_MM, on_warning=None):
        self.robot, self.start, self.end = robot, list(start), list(end)
        self.limit_mm, self.also, self.worst_mm = limit_mm, also, 0.0
        self.warn_mm, self.on_warning, self.warned = min(warn_mm, limit_mm), on_warning, False

    def __call__(self):
        if self.also is not None:
            self.also()
        feedback = self.robot.get_flange_pose()
        if feedback is None:
            return  # The motion loop itself rejects missing or stale feedback.
        deviation = segment_distance_mm(list(feedback.msg)[:3], self.start, self.end)
        self.worst_mm = max(self.worst_mm, deviation)
        if deviation > self.limit_mm:
            raise RuntimeError("플랜지가 직선에서 {:.1f}mm 벗어났습니다 (정지 한도 {:g}mm).".format(
                deviation, self.limit_mm))
        if deviation > self.warn_mm and not self.warned:
            self.warned = True
            print("주의: 플랜지가 직선에서 {:.1f}mm 벗어났습니다. {:g}mm까지는 멈추지 않고 계속합니다.".format(
                deviation, self.limit_mm))
            if self.on_warning is not None:
                self.on_warning(deviation)


def joint_lead_degrees(speed_percent):
    """How far ahead the joint target is kept: about 0.15 s of travel, at least 2 degrees."""
    return max(2.0, 0.3 * speed_percent)


def execute_plan(robot, plan, *, speed_percent=5, timeout=60.0, max_deviation_mm=50.0, recorder=None,
                 joints_together=True, joint_update_hz=100.0):
    """Run a plan on an already enabled arm. Never enables, disables or resets."""
    check = None if recorder is None else recorder.check
    result = dict(max_line_deviation_mm=None, endpoint_error_mm=None)
    if plan.needs_joint_move:
        print("관절 동시 이동을 시작합니다. 속도 비율 {}%.".format(speed_percent))
        if recorder is not None:
            recorder.set_phase(1, "joint_move", plan.entry_pose)
            recorder.event("joint_move_requested", target_rad=plan.entry_joints)
        actual = move_joint_target(robot, plan.entry_joints, timeout=timeout,
                                   max_step_degrees=min(360.0, plan.joint_move_deg + 1.0),
                                   on_feedback=check, speed_percent=speed_percent, allow_long_move=True,
                                   lead_degrees=joint_lead_degrees(speed_percent) if joints_together else None,
                                   update_hz=joint_update_hz)
        if recorder is not None:
            recorder.event("joint_move_completed", actual_rad=actual)
    if len(plan.line_joints) < 2:
        return result
    start, _ = flange_pose(robot)
    gap_mm = math.dist(start[:3], plan.entry_pose[:3]) * 1000.0
    if gap_mm > ENTRY_TOLERANCE_MM:
        raise RuntimeError("직선 시작점이 계획과 {:.1f}mm 다릅니다 (허용 {:g}mm). 직선 이동을 보내지 않습니다.".format(
            gap_mm, ENTRY_TOLERANCE_MM))
    print("직선 이동을 시작합니다.")
    def warned(deviation):
        diagnose(robot, "line_deviation_warning", deviation_mm=deviation, limit_mm=max_deviation_mm)
        if recorder is not None:
            recorder.event("line_deviation_warning", deviation_mm=deviation, limit_mm=max_deviation_mm)

    monitor = LineMonitor(robot, start[:3], plan.target_pose[:3], max_deviation_mm, check, on_warning=warned)
    if recorder is not None:
        recorder.set_phase(1, "line", plan.target_pose)
        recorder.event("line_requested", start_m=start[:3], target_m_rad=plan.target_pose)
    # Right after our own joint move the arm is known to be at rest: skip the
    # 0.5 s stillness window so the two parts follow each other quickly.
    actual = move_linear_target(robot, plan.target_pose, timeout=timeout, on_feedback=monitor,
                                speed_percent=speed_percent, settled=plan.needs_joint_move)
    result.update(max_line_deviation_mm=monitor.worst_mm,
                  endpoint_error_mm=math.dist(actual[:3], plan.target_pose[:3]) * 1000.0)
    if recorder is not None:
        recorder.segments.append(dict(cycle=1, phase="line", actual_m_rad=actual,
                                      target_m_rad=plan.target_pose, **result))
        recorder.event("line_completed", **result)
    return result


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channel", default="0", help="Windows CAN 채널 (기본: 0)")
    parser.add_argument("--x", type=float, help="베이스 기준 절대 X좌표(mm). 생략하면 현재 값 유지")
    parser.add_argument("--y", type=float, help="베이스 기준 절대 Y좌표(mm)")
    parser.add_argument("--z", type=float, help="베이스 기준 절대 Z좌표(mm)")
    parser.add_argument("--orient", choices=("free", "keep", "down"),
                        help="목표 방향. free(기본): 계획기가 선택 / keep: 현재 방향 유지 / down: 도구 축이 아래를 향함")
    parser.add_argument("--rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
                        help="목표 방향을 도(deg)로 직접 지정")
    parser.add_argument("--tool-axis", choices=sorted(TOOL_AXES), default="x",
                        help="--orient down에서 아래로 향할 플랜지 축 (기본 x: 모델상 팔뚝 방향)")
    parser.add_argument("--path", choices=("line", "joint"), default="line",
                        help="line(기본): 플랜지 직선 / joint: 관절 동시 이동 한 번(곡선 경로)")
    parser.add_argument("--execute", action="store_true", help="실제 이동. 생략하면 계획만 표시")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED_PERCENT,
                        help="속도 비율(%%), 1~{}. 기본 {}".format(MAX_SPEED_PERCENT, DEFAULT_SPEED_PERCENT))
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="목표에 가까워지지 않은 채 기다리는 최대 시간(초). 기본 60")
    parser.add_argument("--max-joint-move-deg", type=float, default=360.0,
                        help="관절 동시 이동에서 허용할 최대 관절 변화(도), 0~360. 기본 360(사실상 제한 없음). "
                             "큰 이동을 막고 싶을 때만 지정")
    parser.add_argument("--max-deviation-mm", type=float, default=50.0,
                        help="직선 이동 중 이 거리 넘게 벗어나면 급정지 요청(5~100). 기본 50. "
                             "20mm부터는 경고만 하고 계속 이동")
    parser.add_argument("--min-z-mm", type=float, default=50.0,
                        help="팔꿈치·손목·플랜지 원점의 최저 허용 높이(mm). 기본 50. 충돌 검사는 아님")
    parser.add_argument("--min-sigma", type=float, default=0.05,
                        help="직선 구간의 특이점 여유 한도. 키우면 직선이 특이점에서 더 멀리서 시작")
    parser.add_argument("--joint-move", choices=("together", "single"), default="together",
                        help="together(기본): 관절들이 같은 비율로 움직여 함께 도착 / "
                             "single: 목표를 한 번에 보냄(관절마다 도착 시각이 다름)")
    parser.add_argument("--joint-update-hz", type=float, default=100.0,
                        help="together 방식에서 관절 목표를 갱신하는 빈도(Hz), 10~200. 기본 100. "
                             "20은 2026-11-16 실물에서 움직임이 고르지 않았던 값")
    parser.add_argument("--rotate-on-line", action="store_true",
                        help="직선 이동 중에 방향도 함께 바꿈. 기본은 직선 구간에서 방향 고정. "
                             "펌웨어 1.12에서 NO_SOLUTION으로 실패한 기록이 있음")
    parser.add_argument("--output-dir", default="data/nero_goto", help="실행 기록 폴더의 상위 경로")
    return parser


def main():
    args = get_parser().parse_args()
    robot = recorder = diagnostics = None
    status, reason, code = "completed", "", 0
    try:
        settings = validate_options(args)
        print("기준: 로봇 베이스 / 위치 단위: mm / 제어점: 플랜지(도구 장착면)")
        print("실제 이동 모드입니다. 팔과 도구가 움직일 공간을 비우세요." if args.execute
              else "미리보기: 상태 조회와 경로 계산만 실행합니다.")
        diagnostics = SessionDiagnostics()
        diagnostics.event(None, "start", command="goto", options=vars(args))
        robot, version, profile = connect_nero(args.channel)
        robot._nero_diagnostics = diagnostics
        time.sleep(1.0)
        # Executing requires an enabled, stationary arm; a preview only needs fresh angles.
        joints = check_before_move(robot) if args.execute else read_joints(robot)
        pose, _ = wait_for_flange_pose(robot)
        print("현재 관절각(deg):", _deg(joints))
        print("현재 XYZ(mm):", _mm(pose[:3]))
        gap_mm, gap_deg = model_gap(joints, pose)
        print("모델과 플랜지 피드백 차이: {:.2f}mm / {:.2f}도".format(gap_mm, gap_deg))
        if gap_mm > MODEL_TOLERANCE_MM or gap_deg > MODEL_TOLERANCE_DEG:
            message = ("SDK 운동학 모델과 실제 플랜지 피드백이 허용치({:g}mm/{:g}도)보다 다릅니다. "
                       "TCP 설정이나 영점을 확인하세요.".format(MODEL_TOLERANCE_MM, MODEL_TOLERANCE_DEG))
            if args.execute:
                raise RuntimeError(message + " 이동하지 않습니다.")
            print("주의:", message, "아래 계획은 참고용입니다.")
        target = [pose[i] if value is None else value / 1000.0
                  for i, value in enumerate((args.x, args.y, args.z))]
        planner = plan_joint_move if args.path == "joint" else plan_line
        plan = planner(joints, target, robot.get_config()["joint_limits"],
                       settings=settings, **orientation_request(args))
        describe(plan, settings)
        summary = {k: v for k, v in asdict(plan).items() if k != "line_joints"}
        diagnostics.event(robot, "planned", line_points=len(plan.line_joints), **summary)
        too_far = plan.needs_joint_move and plan.joint_move_deg > args.max_joint_move_deg
        if too_far:
            print("관절 동시 이동이 {:.1f}도로 한도 {:g}도보다 큽니다. 계획을 검토한 뒤 "
                  "--max-joint-move-deg로 한도를 명시해야 실행할 수 있습니다.".format(
                      plan.joint_move_deg, args.max_joint_move_deg))
        if not args.execute:
            print("미리보기 완료. 이동 명령은 보내지 않았습니다. 실행하려면 같은 옵션에 --execute를 붙이세요.")
            return 0
        if too_far:
            raise RuntimeError("관절 이동 한도 초과. 이동하지 않습니다.")
        if not plan.needs_joint_move and plan.straight_mm < 0.5 and plan.orientation_change_deg < 0.5:
            print("이미 목표 위치·방향입니다. 이동 명령을 보내지 않습니다.")
            return 0
        print("{:g}초 후 이동을 시작합니다. 중단하려면 Ctrl+C.".format(START_DELAY_S))
        time.sleep(START_DELAY_S)
        now = datetime.now().astimezone()
        directory = Path(args.output_dir) / (now.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        recorder = RunRecorder(robot, directory, dict(
            schema_version=1, created_at=now.isoformat(), mode="goto", firmware=version,
            sdk_profile=str(profile), options=vars(args), coordinate_frame="robot_base",
            control_point="flange", units="CSV: mm, deg, s; plan: m, rad", plan=asdict(plan),
            speed_percent=args.speed,
            notes="Plan is a model prediction; the controller chooses the real joint path."),
            hz=50, enforce_motion_state=True)  # 50 Hz: enough to see uneven joint speed afterwards.
        print("기록 폴더:", directory.resolve())
        recorder.start()
        result = execute_plan(robot, plan, speed_percent=args.speed, timeout=args.timeout,
                              max_deviation_mm=args.max_deviation_mm, recorder=recorder,
                              joints_together=settings.joints_together,
                              joint_update_hz=args.joint_update_hz)
        if result["max_line_deviation_mm"] is not None:
            print("직선 구간 최대 이탈 {:.2f}mm / 도착 위치 오차 {:.3f}mm (로봇 피드백 기준)".format(
                result["max_line_deviation_mm"], result["endpoint_error_mm"]))
        print("이동 완료. 모터는 활성화 상태로 남습니다.")
        diagnostics.event(robot, "completed", **result)
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        reason, code = str(exc) or "사용자 중단", 130 if isinstance(exc, KeyboardInterrupt) else 1
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=reason)
        print("이동 중단:", reason)
        print("자동 복귀·재시도·reset은 하지 않습니다.")
    finally:
        try:
            if recorder is not None:
                if recorder.finish(status, reason)["status"] == "failed":
                    code = code or 1
        except Exception as exc:
            print("기록 마무리 실패:", exc)
            code = code or 1
        finally:
            try:
                if robot is not None:
                    robot.disconnect()
            finally:
                if diagnostics is not None:
                    diagnostics.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
