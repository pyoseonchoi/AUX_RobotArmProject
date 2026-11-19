"""Stage 0: preview, record only, or repeat a short A-B-A path in free space."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

from nero_enable import DRIVER_ERROR_FIELDS
from nero_move_joint import REJECTIONS, check_before_move
from nero_move_linear import flange_pose, move_linear_target
from nero_move_to import make_target, wait_for_flange_pose
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, require_feedback


POSE_PARTS = ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
JOINT_PARTS = ("joint_12", "joint_34", "joint_56", "joint_7")
FIELDS = (
    ["elapsed_s", "host_epoch_s", "cycle", "phase", "feedback_valid", "error"]
    + ["joint_epoch_s", "pose_epoch_s", "status_epoch_s", "max_feedback_age_s"]
    + [name + "_epoch_s" for name in JOINT_PARTS + POSE_PARTS]
    + ["joint_is_new", "pose_is_new", "ctrl_mode", "arm_status", "motion_status"]
    + ["j{}_enabled".format(i) for i in range(1, 8)]
    + ["j{}_deg".format(i) for i in range(1, 8)]
    + ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
    + ["target_x_mm", "target_y_mm", "target_z_mm", "estimated_speed_mm_s"]
)


def build_path(start, axis="x", mm=5.0):
    """Anchor both endpoints once; never accumulate relative position errors."""
    if axis not in ("x", "y", "z") or not math.isfinite(mm) or not 1 <= abs(mm) <= 20:
        raise ValueError("축은 x/y/z, 편도 거리는 절댓값 1~20mm로 지정하세요.")
    xyz = [None] * 3
    index = ("x", "y", "z").index(axis)
    xyz[index] = start[index] * 1000 + mm
    target, _ = make_target(start, xyz, max_distance_mm=20)
    return list(start), target


def read_sample(robot):
    """Read cached CAN feedback only, including timestamps of component packets."""
    if not robot.is_connected():
        raise FeedbackUnavailableError("CAN 연결이 닫혔습니다.")
    error = robot.get_comm_error()
    if error is not None:
        raise FeedbackUnavailableError("CAN 통신 오류: {}".format(error))
    joint = robot.get_joint_angles()
    pose = robot.get_flange_pose()
    status = robot.get_arm_status()
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    required = [("관절", joint), ("플랜지", pose), ("상태", status)]
    required += [("J{} 드라이버".format(i), d) for i, d in enumerate(drivers, 1)]
    parser = getattr(robot, "_parser", None)
    parts = [] if parser is None else [(name, getattr(parser, name, None))
                                      for name in JOINT_PARTS + POSE_PARTS]
    required += parts
    # Freeze timestamps before reading the clock. Feedback objects are shared
    # with the SDK receive thread; this is not an atomic hardware snapshot.
    stamps = [(name, None if item is None else item.timestamp) for name, item in required]
    require_feedback([(name, None if stamp is None else SimpleNamespace(timestamp=stamp))
                      for name, stamp in stamps])
    angles, position = list(joint.msg), list(pose.msg)
    if len(angles) != 7 or len(position) != 6 or not all(
            math.isfinite(v) for v in angles + position):
        raise RuntimeError("유효한 관절각 7개 / 위치·자세 6개가 필요합니다.")
    now = feedback_time()
    row = {
        "host_epoch_s": now, "joint_epoch_s": stamps[0][1],
        "pose_epoch_s": stamps[1][1], "status_epoch_s": stamps[2][1],
        "max_feedback_age_s": max(now - stamp for _, stamp in stamps),
        "ctrl_mode": int(status.msg.ctrl_mode), "arm_status": int(status.msg.arm_status),
        "motion_status": int(status.msg.motion_status),
    }
    row.update((name + "_epoch_s", stamp) for name, stamp in stamps if name in JOINT_PARTS + POSE_PARTS)
    for i, (angle, driver) in enumerate(zip(angles, drivers), 1):
        row["j{}_deg".format(i)] = math.degrees(angle)
        row["j{}_enabled".format(i)] = int(driver.msg.foc_status.driver_enable_status)
    for name, value in zip(("x_mm", "y_mm", "z_mm"), position[:3]):
        row[name] = value * 1000
    for name, value in zip(("roll_deg", "pitch_deg", "yaw_deg"), position[3:]):
        row[name] = math.degrees(value)
    # Recording alone also works while disabled. Executed runs additionally
    # enforce normal/enabled/no-error feedback in the recorder and motion API.
    # A planning refusal (NO_SOLUTION etc.) is left to the motion API, which
    # can tell a refusal of the current target from an older leftover flag.
    healthy = (row["ctrl_mode"] == 1 and row["arm_status"] in (0,) + REJECTIONS
               and row["motion_status"] in (0, 1)
               and all(row["j{}_enabled".format(i)] for i in range(1, 8))
               and not any(getattr(d.msg.foc_status, name)
                           for d in drivers for name in DRIVER_ERROR_FIELDS))
    return row, healthy


class RunRecorder:
    """One read-only polling thread owns the CSV; motion checks its health."""

    def __init__(self, robot, directory, metadata, hz=20, enforce_motion_state=False):
        self.robot, self.hz, self.enforce_motion_state = robot, hz, enforce_motion_state
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.context = {"cycle": 0, "phase": "baseline"}
        self.failure = None
        self.last_sample_at = None
        self.rows = self.valid_rows = self.unique_poses = 0
        self.segments = []
        self.thread = None
        self.events = None
        (self.directory / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.events = (self.directory / "events.jsonl").open("w", encoding="utf-8")

    def event(self, name, **values):
        values.update(event=name, elapsed_s=time.monotonic() - self.started,
                      host_epoch_s=feedback_time())
        self.events.write(json.dumps(values, ensure_ascii=False) + "\n")
        self.events.flush()

    def set_phase(self, cycle, phase, target=None):
        with self.lock:
            self.context = {"cycle": cycle, "phase": phase}
            if target is not None:
                self.context.update(zip(("target_x_mm", "target_y_mm", "target_z_mm"),
                                        [v * 1000 for v in target[:3]]))
        self.event("phase", cycle=cycle, phase=phase)

    def start(self):
        self.thread = threading.Thread(target=self._record, name="nero-air-recorder", daemon=True)
        self.thread.start()
        if not self.ready.wait(2):
            raise RuntimeError("기록 스레드가 2초 안에 시작되지 않았습니다.")
        self.check()

    def check(self):
        if self.failure is not None:
            raise RuntimeError("기록 중단: {}".format(self.failure)) from self.failure
        stamp = self.last_sample_at
        age = None if stamp is None else time.monotonic() - stamp
        if age is None or age >= 0.5:
            raise RuntimeError(
                "0.5초 안에 기록 샘플을 확인하지 못했습니다. 마지막 저장 후 {}초, "
                "저장 행 수={}, 기록 스레드 실행={}".format(
                    "미수신" if age is None else "{:.3f}".format(age), self.rows,
                    self.thread is not None and self.thread.is_alive()))

    def _record(self):
        previous = None
        previous_joint_stamp = None
        due = time.monotonic()
        try:
            with (self.directory / "robot.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                while not self.stop.is_set():
                    with self.lock:
                        row = dict(self.context)
                    row.update(elapsed_s=time.monotonic() - self.started,
                               host_epoch_s=feedback_time(), feedback_valid=0)
                    try:
                        sample, healthy = read_sample(self.robot)
                        row.update(sample, feedback_valid=1, error="")
                        if self.enforce_motion_state and not healthy:
                            raise RuntimeError(
                                "모터 활성화 / 정상 CAN 상태 / 드라이버 오류를 확인하세요. "
                                "(ctrl_mode={}, arm_status={}, 활성화={})".format(
                                    sample["ctrl_mode"], sample["arm_status"],
                                    [sample["j{}_enabled".format(i)] for i in range(1, 8)]))
                        row["joint_is_new"] = int(sample["joint_epoch_s"] != previous_joint_stamp)
                        previous_joint_stamp = sample["joint_epoch_s"]
                        is_new = previous is None or sample["pose_epoch_s"] != previous["pose_epoch_s"]
                        row["pose_is_new"] = int(is_new)
                        if is_new and previous is not None:
                            dt = sample["pose_epoch_s"] - previous["pose_epoch_s"]
                            if 0 < dt < 0.5:
                                row["estimated_speed_mm_s"] = math.sqrt(sum(
                                    (sample[k] - previous[k]) ** 2 for k in ("x_mm", "y_mm", "z_mm"))) / dt
                        if is_new:
                            self.unique_poses += 1
                            previous = sample
                    except Exception as exc:
                        row.update(feedback_valid=0, error=str(exc))
                        previous = None
                        previous_joint_stamp = None
                        if self.enforce_motion_state:
                            self.failure = exc
                    writer.writerow(row)
                    stream.flush()
                    self.rows += 1
                    self.valid_rows += row["feedback_valid"]
                    self.last_sample_at = time.monotonic()
                    self.ready.set()
                    if self.failure is not None:
                        break
                    # No catch-up bursts: the requested rate is approximate.
                    due = max(due + 1.0 / self.hz, time.monotonic())
                    self.stop.wait(max(0, due - time.monotonic()))
        except Exception as exc:
            self.failure = exc
        finally:
            self.ready.set()

    def finish(self, status, reason=""):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("기록 스레드 종료를 확인하지 못했습니다.")
        if self.failure is not None:
            status, reason = "failed", str(self.failure)
        duration = time.monotonic() - self.started
        position_errors = [s["endpoint_error_mm"] for s in self.segments
                           if s.get("endpoint_error_mm") is not None]
        joint_errors = [s["max_joint_error_deg"] for s in self.segments
                        if s.get("max_joint_error_deg") is not None]
        summary = {
            "status": status, "reason": reason, "duration_s": duration,
            "rows": self.rows, "valid_rows": self.valid_rows,
            "invalid_rows": self.rows - self.valid_rows, "unique_pose_samples": self.unique_poses,
            "observed_poll_hz": (self.rows - 1) / duration if duration else 0,
            "completed_segments": self.segments,
            "max_endpoint_error_mm": max(position_errors) if position_errors else None,
            "max_joint_error_deg": max(joint_errors) if joint_errors else None,
        }
        try:
            self.event("finished", status=status, reason=reason)
            (self.directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            self.events.close()
        return summary


def observe(recorder, seconds):
    deadline = time.monotonic() + seconds
    while True:
        recorder.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def repeat_path(robot, recorder, a, b, cycles=3, pause=1.0, timeout=30.0):
    observe(recorder, 1.0)
    for cycle in range(1, cycles + 1):
        print("왕복 {}/{}".format(cycle, cycles))
        for phase, target in (("outbound", b), ("return", a)):
            recorder.set_phase(cycle, phase, target)
            recorder.check()
            current, _ = flange_pose(robot)
            # A millimeter of allowance covers the existing 0.5 mm arrival
            # tolerance at the endpoint; it is not a workspace/IK check.
            make_target(current, [v * 1000 for v in target[:3]], max_distance_mm=21)
            recorder.event("segment_requested", cycle=cycle, phase=phase, target_m_rad=target)
            actual = move_linear_target(robot, target, timeout=timeout, on_feedback=recorder.check)
            error = math.sqrt(sum((x - y) ** 2 for x, y in zip(actual[:3], target[:3]))) * 1000
            result = dict(cycle=cycle, phase=phase, endpoint_error_mm=error,
                          actual_m_rad=actual, target_m_rad=target)
            recorder.segments.append(result)
            recorder.event("segment_completed", **result)
            recorder.set_phase(cycle, "hold_b" if phase == "outbound" else "hold_a", target)
            observe(recorder, pause)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="경로 표시만 수행 (기본). 이동하지 않음")
    modes.add_argument("--record-only", action="store_true", help="이동 없이 상태를 파일로 기록")
    modes.add_argument("--execute", action="store_true", help="실제 공중 왕복 실행")
    parser.add_argument("--channel", default="0")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--mm", type=float, default=5.0, help="베이스 축 방향 편도 거리, 부호 있는 1~20mm")
    parser.add_argument("--cycles", type=int, default=3, help="왕복 횟수, 1~100")
    parser.add_argument("--pause", type=float, default=1.0, help="각 도착 후 대기 시간(초)")
    parser.add_argument("--timeout", type=float, default=30.0, help="편도 이동 완료 대기 시간(초)")
    parser.add_argument("--duration", type=float, default=10.0, help="기록 전용 모드의 시간(초)")
    parser.add_argument("--sample-hz", type=float, default=20.0, help="기록 폴링 목표 Hz, 5~50")
    parser.add_argument("--output-dir", default="data/nero_air", help="실험 폴더를 생성할 상위 경로")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    robot = recorder = None
    status, reason, code = "completed", "", 0
    try:
        if not 1 <= args.cycles <= 100:
            raise ValueError("왕복 횟수는 1~100입니다.")
        if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.duration)):
            raise ValueError("timeout/duration은 0보다 큰 유한한 초 값이어야 합니다.")
        if not math.isfinite(args.pause) or args.pause < 0:
            raise ValueError("pause는 0 이상의 유한한 초 값이어야 합니다.")
        if not math.isfinite(args.sample_hz) or not 5 <= args.sample_hz <= 50:
            raise ValueError("기록 주파수는 5~50Hz입니다.")
        build_path([0.0] * 6, args.axis, args.mm)  # Reject bad options before connecting.
        mode = "execute" if args.execute else "record_only" if args.record_only else "preview"
        print("0단계 모드: {} / 베이스 좌표계 / 플랜지 기준".format(mode))
        robot, version, profile = connect_nero(args.channel)
        time.sleep(1.0)
        start, _ = wait_for_flange_pose(robot)
        if args.execute:
            check_before_move(robot)
            start, _ = flange_pose(robot)
        a, b = build_path(start, args.axis, args.mm)
        if not args.record_only:
            print("A 현재 XYZ(mm):", [round(v * 1000, 3) for v in a[:3]])
            print("B 목표 XYZ(mm):", [round(v * 1000, 3) for v in b[:3]])
            print("A → B → A, {}회. 방향 유지 / 속도 비율 5%.".format(args.cycles))
            print("도달 가능성·충돌·접촉 여부를 자동 검사하지 않습니다.")
        if mode == "preview":
            print("미리보기 완료. 모터 활성화·이동 명령과 파일 저장은 하지 않았습니다.")
            return 0
        now = datetime.now().astimezone()
        directory = Path(args.output_dir) / (now.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        metadata = dict(schema_version=1, created_at=now.isoformat(), mode=mode,
                        firmware=version, sdk_profile=str(profile), options=vars(args),
                        coordinate_frame="robot_base", control_point="flange",
                        units="CSV: mm, deg, s; poses in JSON: m, rad",
                        a_m_rad=a, b_m_rad=b, speed_percent=5 if args.execute else None,
                        notes="Polled cached feedback; packets are asynchronous. Speed is an estimate. No audio or force recorded.")
        recorder = RunRecorder(robot, directory, metadata, hz=args.sample_hz,
                               enforce_motion_state=args.execute)
        print("기록 폴더:", directory.resolve())
        recorder.start()
        if args.execute:
            print("실제 공중 왕복을 시작합니다. 자동 enable/reset/disable은 하지 않습니다.")
            repeat_path(robot, recorder, a, b, args.cycles, args.pause, args.timeout)
            print("왕복 완료. 모터는 활성화 상태로 남습니다.")
        else:
            recorder.set_phase(0, "record_only")
            observe(recorder, args.duration)
            print("상태 기록 완료. 모터 활성화·이동 명령은 보내지 않았습니다.")
    except (Exception, KeyboardInterrupt) as exc:
        status, reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc) or "사용자 중단"
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print("0단계 중단:", reason)
        print("자동 복귀·재시도·reset은 하지 않습니다.")
    finally:
        try:
            if recorder is not None:
                summary = recorder.finish(status, reason)
                print("기록 행: {} / 유효: {} / 무효: {}".format(
                    summary["rows"], summary["valid_rows"], summary["invalid_rows"]))
                print("실측 기록 빈도: {:.1f}Hz / 완료한 편도 이동: {}회".format(
                    summary["observed_poll_hz"], len(summary["completed_segments"])))
                if summary["completed_segments"]:
                    print("최대 도착 위치 오차: {:.3f}mm (로봇 피드백 기준)".format(
                        summary["max_endpoint_error_mm"]))
                print("저장:", recorder.directory.resolve())
                if summary["status"] == "failed":
                    code = 1
        except Exception as exc:
            print("기록 마무리 실패:", exc)
            code = 1
        finally:
            if robot is not None:
                robot.disconnect()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
