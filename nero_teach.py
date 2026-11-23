"""Record hand-guided Nero joint paths, inspect them, return home and replay."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
from nero_enable import DRIVER_ERROR_FIELDS
from nero_kinematics import height_floor_mm, height_limit_text, joint_move_trace
from nero_move_joint import check_before_move, follow_joint_path, motion_feedback, move_joint_target
from nero_move_linear import flange_pose
from nero_read_status import FeedbackUnavailableError, connect_nero, feedback_time, recent, require_feedback
from nero_repeat_air import RunRecorder, observe


FORMAT = "nero_joint_teaching_v1"
LEADER_PARTS = ("leader_joint_12", "leader_joint_34", "leader_joint_56", "leader_joint_7")
NORMAL_PARTS = ("joint_12", "joint_34", "joint_56", "joint_7")
RAD_KEYS = ["j{}_rad".format(i) for i in range(1, 8)]
START_TOLERANCE = math.radians(0.5)
MIN_WAYPOINT_DEG = 0.3  # Simplified points closer than this to the previous one are merged.
DEFAULT_SPEED_PERCENT = 7  # Same default and upper bound as nero_goto.py / nero_home.py.
MAX_SPEED_PERCENT = 20
START_DELAY_S = 3.0  # Time to abort with Ctrl+C before a direct return starts.
DIAGNOSTIC_ROOT = Path("data/nero_teach_logs")


class SessionDiagnostics:
    """Persist CLI control/entry failures too, not just completed recordings."""
    def __init__(self):
        DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = DIAGNOSTIC_ROOT / (datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_")
                                       + uuid.uuid4().hex[:8] + ".jsonl")
        self.stream = self.path.open("x", encoding="utf-8")

    def event(self, robot, event, **values):
        row = dict(event=event, time=datetime.now().astimezone().isoformat(), **values)
        if robot is not None:
            try:
                status = robot.get_arm_status()
                row["status_fresh"] = recent(status)
                if status is not None:
                    row["status_age_s"] = feedback_time() - status.timestamp
                    for name in ("ctrl_mode", "arm_status", "mode_feedback", "motion_status"):
                        value = getattr(status.msg, name)
                        row[name] = str(value)
                        row[name + "_value"] = int(value)
                drivers = [robot.get_driver_states(i) for i in range(1, 8)]
                row["driver_feedback_fresh"] = [recent(d) for d in drivers]
                row["enabled"] = [None if d is None else bool(d.msg.foc_status.driver_enable_status)
                                  for d in drivers]
            except Exception as exc:
                row["snapshot_error"] = str(exc)
        try:
            self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.stream.flush()
        except OSError as exc:
            # Diagnostic disk failures must not block an already-needed exit.
            print("진단 로그 저장 실패:", exc)

    def close(self):
        self.stream.close()


def _diagnose(robot, event, **values):
    logger = getattr(robot, "_nero_teach_diagnostics", None)
    if logger is not None:
        logger.event(robot, event, **values)


def connect_known_nero(channel, profile):
    """Recovery only: the operator supplies a previously verified firmware profile.

    A controller with reporting disabled may not answer firmware queries either.
    Connecting with a known profile sends no configuration or motion commands.
    """
    robot = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=profile, interface="agx_cando", channel=channel))
    try:
        robot.connect()
    except Exception:
        robot.disconnect()
        raise
    return robot, "조회 생략 (확인한 버전 사용)", profile


def joint_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def validate_joints(values, limits):
    if len(values) != 7 or not all(isinstance(x, (float, int)) and not isinstance(x, bool)
                                   and math.isfinite(x) for x in values):
        raise ValueError("관절각은 유한한 rad 숫자 7개여야 합니다.")
    for i, value in enumerate(values, 1):
        low, high = limits["joint{}".format(i)]
        if not low <= value <= high:
            raise ValueError("J{} 각도가 현재 SDK 제한 범위 밖입니다.".format(i))


def _check_reported_faults(robot):
    if not robot.is_connected() or robot.get_comm_error() is not None:
        raise RuntimeError("교시 기록 중 CAN 연결/통신 오류가 발생했습니다.")
    status = robot.get_arm_status()
    if recent(status) and int(status.msg.arm_status) in (1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 15):
        raise RuntimeError("교시 중 로봇 오류 상태입니다: {}".format(status.msg.arm_status))
    for i in range(1, 8):
        driver = robot.get_driver_states(i)
        if recent(driver) and any(getattr(driver.msg.foc_status, name) for name in DRIVER_ERROR_FIELDS):
            raise RuntimeError("교시 중 J{} 드라이버 오류가 보고됐습니다.".format(i))


def leader_sample(robot, *, newer_than=None):
    """Require every leader packet; never substitute normal feedback in a recording."""
    _check_reported_faults(robot)
    feedback = robot.get_leader_joint_angles()
    parser = getattr(robot, "_parser", None)
    parts = [getattr(parser, name, None) for name in LEADER_PARTS]
    stamps = [None if item is None else item.timestamp for item in parts]
    frozen = [(name, None if stamp is None else SimpleNamespace(timestamp=stamp))
              for name, stamp in zip(LEADER_PARTS, stamps)]
    require_feedback(frozen, newer_than=newer_than)
    if feedback is None:
        raise FeedbackUnavailableError("교시 관절각 통합값이 없습니다.")
    joints = list(feedback.msg)
    validate_joints(joints, robot.get_config()["joint_limits"])
    return joints, stamps


def normal_sample(robot, *, newer_than=None):
    """Normal telemetry can observe a supported arm even if entry was rejected."""
    _check_reported_faults(robot)
    feedback = robot.get_joint_angles()
    parser = getattr(robot, "_parser", None)
    parts = [getattr(parser, name, None) for name in NORMAL_PARTS]
    stamps = [None if item is None else item.timestamp for item in parts]
    required = list(zip(NORMAL_PARTS, parts))
    required += [("일반 상태", robot.get_arm_status())]
    required += [("J{} 드라이버".format(i), robot.get_driver_states(i)) for i in range(1, 8)]
    frozen = [(name, None if item is None else SimpleNamespace(timestamp=item.timestamp)) for name, item in required]
    require_feedback(frozen, newer_than=newer_than)
    if feedback is None:
        raise FeedbackUnavailableError("일반 관절각 통합값이 없습니다.")
    joints = list(feedback.msg)
    validate_joints(joints, robot.get_config()["joint_limits"])
    return joints, stamps


def linkage_sample(robot, *, newer_than=None):
    """Firmware may report ordinary joint packets while explicitly in mode 6."""
    joints, stamps = normal_sample(robot, newer_than=newer_than)
    status = robot.get_arm_status()
    if int(status.msg.ctrl_mode) != 6:
        raise FeedbackUnavailableError("일반 관절 피드백은 있지만 연동 교시 모드(6)가 아닙니다: {}".format(status.msg.ctrl_mode))
    if int(status.msg.arm_status) not in (0, 11):
        raise RuntimeError("교시 기록을 시작할 정상 상태가 아닙니다: {}".format(status.msg.arm_status))
    if not all(robot.get_driver_states(i).msg.foc_status.driver_enable_status for i in range(1, 8)):
        raise RuntimeError("교시 모드에서 모터 활성화를 확인하지 못했습니다.")
    return joints, stamps


def wait_for_teaching(robot, *, newer_than=None, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        reasons = []
        for reader, source in ((leader_sample, "leader_packets"),
                               (linkage_sample, "normal_joint_packets_in_linkage_mode")):
            try:
                joints, stamps = reader(robot, newer_than=newer_than)
                return reader, source, joints, stamps
            except FeedbackUnavailableError as exc:
                reasons.append(str(exc))
        if time.monotonic() >= deadline:
            raise FeedbackUnavailableError("교시 피드백을 확인하지 못했습니다: " + " / ".join(reasons))
        time.sleep(0.02)


SOURCE_NAMES = {}  # Filled below: reader function -> name stored in the recording.


class TeachingFeed:
    """Teaching joint angles from whichever stream the firmware is sending now.

    Leader packets are always preferred; ordinary joint packets are used only
    while they still report linkage mode 6. On 2026-11-19 13:18 firmware 1.12
    first answered the teaching request on the ordinary packets with mode 6,
    then 0.5 s later stopped every ordinary joint, status and driver packet
    (switching to leader packets, as on 2026-11-18). The recorder had locked
    onto the ordinary packets, so both the recording and the exit to control
    failed and the arm was left in teaching mode. Choosing per sample also
    avoids a recording gap when the firmware switches.
    """

    MAX_SWITCH_JUMP = math.radians(10)  # Two streams disagreeing by more than this is not a switch.

    def __init__(self):
        self.active = None
        self.last = None
        self.switches = []
        self.history = []  # Source names in the order they were used.

    def __call__(self, robot, *, newer_than=None):
        reasons = []
        for reader in (leader_sample, linkage_sample):
            try:
                joints, stamps = reader(robot, newer_than=newer_than)
            except FeedbackUnavailableError as exc:
                reasons.append(str(exc))
                continue
            if self.active is not None and reader is not self.active:
                if self.last is not None and joint_error(joints, self.last) > self.MAX_SWITCH_JUMP:
                    raise RuntimeError("교시 데이터 출처가 바뀌는 순간 관절각이 10도 넘게 달라 기록을 중단합니다.")
                self.switches.append(dict(at=feedback_time(), source=SOURCE_NAMES[reader]))
                print("교시 데이터 출처가 바뀌었습니다:", SOURCE_NAMES[reader])
            if reader is not self.active:
                self.history.append(SOURCE_NAMES[reader])
            self.active, self.last = reader, list(joints)
            return joints, stamps
        raise FeedbackUnavailableError("교시 피드백이 없습니다: " + " / ".join(reasons))

    @property
    def source(self):
        return " -> ".join(self.history) if self.history else "unconfirmed"


SOURCE_NAMES.update({leader_sample: "leader_packets", linkage_sample: "normal_joint_packets_in_linkage_mode"})


def _may_lose_normal_feedback(reader):
    """Leader firmware can suppress all ordinary packets; such readers must not need them."""
    return reader is leader_sample or isinstance(reader, TeachingFeed)


def wait_for_leader(robot, *, newer_than=None, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return leader_sample(robot, newer_than=newer_than)
        except FeedbackUnavailableError as exc:
            if time.monotonic() >= deadline:
                raise FeedbackUnavailableError("교시 패킷 4개를 확인하지 못했습니다: {}".format(exc))
            time.sleep(0.02)


def settle_leader(robot, on_sample=None, timeout=10.0, *, reader=leader_sample):
    """Wait for supported/stationary hand guidance before changing mode."""
    deadline = time.monotonic() + timeout
    low = high = None
    stable_since = None
    previous = None
    fresh_count = 0
    while True:
        joints, stamps = reader(robot)
        if previous is None or all(a > b for a, b in zip(stamps, previous)):
            if on_sample is not None:
                on_sample(joints, stamps)
            previous = stamps
            if low is None:
                low, high = list(joints), list(joints)
                stable_since, fresh_count = time.monotonic(), 0
            low = [min(a, b) for a, b in zip(low, joints)]
            high = [max(a, b) for a, b in zip(high, joints)]
            fresh_count += 1
            if joint_error(low, high) > math.radians(0.05):
                low, high = list(joints), list(joints)
                stable_since, fresh_count = time.monotonic(), 0
            if fresh_count >= 3 and time.monotonic() - stable_since >= 0.5:
                return joints
        if time.monotonic() >= deadline:
            raise RuntimeError("팔이 멈춘 것을 확인하지 못했습니다. 팔을 지지하고 control 명령으로 복귀하세요.")
        time.sleep(0.02)


def _enable_normal_feedback(robot):
    """Undo the SDK leader/follower helper's explicit CAN-push disable.

    V112 set_normal_mode() is a no-op and there is no public push-only setter.
    Use the same SDK mode packet with an invalid (unchanged) motion mode; send
    no target, enable, disable or reset. Restore cached command fields afterward.
    """
    mode = robot._msg_mode
    saved_push, saved_move = mode.enable_can_push, mode.move_mode
    saved_speed, saved_mit = mode.move_spd_rate_ctrl, mode.mit_mode
    try:
        mode.enable_can_push = mode.Enums.CanActiveMsgReporting.ENABLE
        mode.move_mode = 255
        mode.move_spd_rate_ctrl = 5
        mode.mit_mode = 255  # Preserve the controller's current motor-control mode.
        robot._set_mode()
    finally:
        mode.enable_can_push, mode.move_mode = saved_push, saved_move
        mode.move_spd_rate_ctrl, mode.mit_mode = saved_speed, saved_mit


def request_teaching_mode(robot):
    """Request the SDK's leader role without first disabling normal CAN push.

    The SDK public helper disables normal reporting before requesting leader.
    If firmware rejects that role, it leaves no telemetry to diagnose/restore.
    The single-arm recorder only needs the role-config packet. Confirmation
    uses fresh leader packets or fresh normal packets with explicit
    linkage mode 6. Neither observation measures gravity compensation quality.
    """
    robot._set_leader_follower_config(linkage_config=0xFA)


def wait_for_normal(robot, *, newer_than=None, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return normal_sample(robot, newer_than=newer_than)
        except FeedbackUnavailableError as exc:
            if time.monotonic() >= deadline:
                raise FeedbackUnavailableError("일반 피드백 재개를 확인하지 못했습니다: {}".format(exc))
            time.sleep(0.02)


def _request_can_control(robot, motion_mode):
    """Request CAN rights with an explicitly chosen planned motion type.

    Recovery selects the pre-teaching P/J/L/C type. Explicit prepare may choose
    J only after confirming all motors disabled. Neither caller sends targets,
    MIT/CPV commands, motor enable, disable or reset. Speed is set to 5%.
    """
    if motion_mode not in (0, 1, 2, 3):
        raise RuntimeError("현재 이동 모드가 P/J/L/C가 아니므로 자동 제어 복귀하지 않습니다.")
    mode = robot._msg_mode
    names = ("ctrl_mode", "move_mode", "mit_mode", "move_spd_rate_ctrl", "enable_can_push")
    saved = [getattr(mode, name) for name in names]
    try:
        mode.ctrl_mode = 1
        mode.move_mode = motion_mode
        mode.mit_mode = 0
        mode.move_spd_rate_ctrl = 5
        mode.enable_can_push = mode.Enums.CanActiveMsgReporting.ENABLE
        robot._set_mode()
    finally:
        for name, value in zip(names, saved):
            setattr(mode, name, value)


def _recovery_enable_state(robot):
    """Allow a fully disabled arm to leave mode 6 without enabling any motor."""
    enabled = tuple(bool(robot.get_driver_states(i).msg.foc_status.driver_enable_status)
                    for i in range(1, 8))
    arm = int(robot.get_arm_status().msg.arm_status)
    if all(enabled) and arm in (0, 11):
        return enabled
    if not any(enabled) and arm in (0, 6):
        return enabled
    raise RuntimeError("제어 복귀 전 모터/팔 상태가 일관되지 않습니다. 전체 활성화 또는 전체 비활성 상태를 확인하세요.")


def wait_for_can_control(robot, reference, *, newer_than, expected_enabled,
                         expected_motion_mode=None, timeout=3.0):
    """Mode 6 can persist during handover; wait rather than reject it immediately."""
    deadline = time.monotonic() + timeout
    last_reason = "새 일반 피드백 대기"
    acknowledged_at = None
    previous_stamps = None
    acknowledged_samples = 0
    while True:
        try:
            joints, stamps = normal_sample(robot, newer_than=newer_than)
            status = robot.get_arm_status()
            if joint_error(joints, reference) > math.radians(1):
                robot.electronic_emergency_stop()
                raise RuntimeError("모드 전환 후 자세가 1도 넘게 달라져 급정지를 요청했습니다.")
            ctrl, arm = int(status.msg.ctrl_mode), int(status.msg.arm_status)
            enabled = tuple(bool(robot.get_driver_states(i).msg.foc_status.driver_enable_status)
                            for i in range(1, 8))
            allowed_arm = (0,) if all(expected_enabled) else (0, 6)
            if arm not in allowed_arm or enabled != expected_enabled:
                raise RuntimeError("제어 복귀 중 팔 상태 또는 모터 활성화 상태가 달라졌습니다.")
            mode = int(status.msg.mode_feedback)
            if ctrl == 1 and (expected_motion_mode is None or mode == expected_motion_mode):
                if acknowledged_at is None:
                    acknowledged_at = time.monotonic()
                if previous_stamps is None or all(a > b for a, b in zip(stamps, previous_stamps)):
                    acknowledged_samples += 1
                    previous_stamps = stamps
                if acknowledged_samples >= 3 and time.monotonic() - acknowledged_at >= 0.2:
                    return joints
            else:
                acknowledged_at, previous_stamps, acknowledged_samples = None, None, 0
            if ctrl not in (1, 6):
                raise RuntimeError("제어 복귀 중 예상하지 못한 제어 모드: {}".format(status.msg.ctrl_mode))
            last_reason = "ctrl_mode={} / arm_status={} / mode_feedback={}".format(
                status.msg.ctrl_mode, status.msg.arm_status, status.msg.mode_feedback)
        except FeedbackUnavailableError as exc:
            acknowledged_at, previous_stamps, acknowledged_samples = None, None, 0
            last_reason = str(exc)
        if time.monotonic() >= deadline:
            raise FeedbackUnavailableError("CAN 제어 복귀 시간 초과: {}".format(last_reason))
        time.sleep(0.02)


def restore_control(robot, *, on_sample=None, reader=None, motion_mode=None,
                    expected_enabled=None, timeout=3.0):
    _check_reported_faults(robot)
    if motion_mode is not None and motion_mode not in (0, 1, 2, 3):
        raise RuntimeError("현재 이동 모드가 P/J/L/C가 아니므로 목표 없이 제어 복귀할 수 없습니다.")
    if reader is None:
        try:
            leader_sample(robot)
            reader = leader_sample
        except FeedbackUnavailableError:
            requested_at = feedback_time()
            _enable_normal_feedback(robot)
            wait_for_normal(robot, newer_than=requested_at)
            reader = normal_sample
            print("교시 전용 패킷이 없어 일반 관절 피드백으로 정지를 관찰합니다.")
    # Real leader firmware may suppress all normal status/driver packets, even
    # while CAN push is enabled. Those packets cannot be a prerequisite for exit.
    try:
        normal_sample(robot)
    except FeedbackUnavailableError:
        if not _may_lose_normal_feedback(reader):
            raise
    else:
        actual_enabled = _recovery_enable_state(robot)
        if expected_enabled is not None and actual_enabled != expected_enabled:
            raise RuntimeError("제어 복귀 전 모터 활성화 상태가 기존 상태와 다릅니다.")
        expected_enabled = actual_enabled
    last = settle_leader(robot, on_sample=on_sample, reader=reader)
    try:
        normal_sample(robot)
    except FeedbackUnavailableError:
        if not _may_lose_normal_feedback(reader):
            raise
    else:
        actual_enabled = _recovery_enable_state(robot)
        if expected_enabled is not None and actual_enabled != expected_enabled:
            raise RuntimeError("정지 관찰 중 모터 활성화 상태가 달라져 역할 전환을 중단합니다.")
        expected_enabled = actual_enabled
    # Use only the role packet; do not disable reporting again on the way out.
    _diagnose(robot, "before_follower_request", expected_motion_mode=motion_mode)
    handover_at = feedback_time()
    robot._set_leader_follower_config(linkage_config=0xFC)
    time.sleep(0.05)
    _enable_normal_feedback(robot)
    current, _ = wait_for_normal(robot, newer_than=handover_at, timeout=timeout)
    _diagnose(robot, "after_follower_feedback", expected_motion_mode=motion_mode)
    if joint_error(current, last) > math.radians(1):
        robot.electronic_emergency_stop()
        raise RuntimeError("역할 전환 후 자세가 1도 넘게 달라져 급정지를 요청했습니다.")
    actual_enabled = _recovery_enable_state(robot)
    if expected_enabled is not None and actual_enabled != expected_enabled:
        raise RuntimeError("역할 전환 중 모터 활성화 상태가 달라져 CAN 제어 요청을 중단합니다.")
    if expected_enabled is None:
        print("follower 전환 후 최신 일반 관절·상태·드라이버 피드백을 확인했습니다.")
    expected_enabled = actual_enabled
    status = robot.get_arm_status().msg
    observed_mode = int(status.mode_feedback)
    # V112 can leave follower handover at ctrl=6 / MOVE_MIT. That is an
    # intermediate teaching state, not an acknowledgement of planned control.
    # Request the saved planned mode only after the fresh normal telemetry,
    # stationary joints and unchanged motor state checks above. A standalone
    # recovery has no saved mode and deliberately selects J. Never generalize
    # this exception to an unrelated CAN/MIT or CPV session.
    teaching_mit = int(status.ctrl_mode) == 6 and observed_mode == 6
    if motion_mode is None:
        motion_mode = 1 if teaching_mit else observed_mode
    if motion_mode not in (0, 1, 2, 3) or (motion_mode != observed_mode and not teaching_mit):
        raise RuntimeError("복귀 후 이동 모드가 기존 P/J/L/C 모드와 일치하지 않아 CAN 제어 요청을 중단합니다.")
    if teaching_mit:
        print("교시 종료 후 MIT 중간 상태를 확인했습니다. 관절 정지 확인 후 일반 제어 모드로 복귀를 요청합니다.")
    _diagnose(robot, "can_control_requested", requested_motion_mode=motion_mode,
              teaching_mit_handover=teaching_mit)
    sent_at = feedback_time()
    print("이동 모드({}) / 속도 5%로 CAN 제어권을 요청합니다. 이동 목표는 보내지 않습니다.".format(motion_mode))
    _request_can_control(robot, motion_mode)
    current = wait_for_can_control(robot, last, newer_than=sent_at,
                                   expected_enabled=expected_enabled,
                                   expected_motion_mode=motion_mode, timeout=timeout)
    print("최신 일반 피드백으로 제어 모드 복귀를 확인했습니다. 이동 목표는 보내지 않았습니다.")
    if not any(expected_enabled):
        print("모터 7개는 비활성 상태를 유지했습니다. 기록하려면 별도로 enable한 뒤 상태를 확인하세요.")
    return current


def prepare_joint_mode(robot, timeout=3.0):
    """Explicit preparation: only change the mode when every motor is disabled."""
    def disabled_sample(arm):
        joints, stamps = normal_sample(arm)
        status = arm.get_arm_status().msg
        if int(status.ctrl_mode) != 1 or int(status.arm_status) not in (0, 6):
            raise RuntimeError("prepare는 일반 CAN 제어의 정상/브레이크 미해제 상태에서만 가능합니다.")
        if any(arm.get_driver_states(i).msg.foc_status.driver_enable_status for i in range(1, 8)):
            raise RuntimeError("prepare는 모터 7개가 모두 비활성일 때만 가능합니다. 팔을 지지하고 별도로 disable하세요.")
        if int(status.mode_feedback) not in (0, 1, 2, 3, 5, 6):
            raise RuntimeError("알 수 없는 이동 모드이므로 prepare하지 않습니다: {}".format(status.mode_feedback))
        return joints, stamps

    print("준비 검사: 모터 7개 비활성 상태에서 0.5초 동안 관절 정지를 확인합니다.")
    wait_for_normal(robot, newer_than=feedback_time(), timeout=timeout)
    reference = settle_leader(robot, reader=disabled_sample)
    disabled_sample(robot)
    print("모터를 끈 상태에서 MOVE_J / 위치·속도 제어 / 속도 비율 5%를 요청합니다. 이동 목표는 보내지 않습니다.")
    sent_at = feedback_time()
    _request_can_control(robot, 1)
    current = wait_for_can_control(robot, reference, newer_than=sent_at,
                                   expected_enabled=(False,) * 7,
                                   expected_motion_mode=1, timeout=timeout)
    print("준비 완료: CAN_CTRL / MOVE_J / 모터 7개 비활성을 확인했습니다. 별도로 enable한 뒤 상태를 확인하세요.")
    return current


class TeachingFile:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata, self.samples = metadata, []
        self.started = time.perf_counter()
        self.previous = None
        self.stream = (self.directory / "joints.csv").open("w", encoding="utf-8-sig", newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["elapsed_s", "host_epoch_s"] + RAD_KEYS
                             + ["j{}_deg".format(i) for i in range(1, 8)]
                             + ["component_{}_epoch_s".format(name) for name in ("12", "34", "56", "7")])
        # An unclean process exit leaves an explicitly non-replayable manifest.
        try:
            self.save("recording", False, "")
        except Exception:
            self.stream.close()
            raise

    def append(self, joints, stamps):
        if self.previous is not None and not all(a > b for a, b in zip(stamps, self.previous)):
            return
        # On Windows/Python 3.12 monotonic() can tick coarsely enough that
        # the first two fresh CAN samples both get t_s=0. Use the high-resolution
        # counter, and never persist a duplicate/backward elapsed timestamp.
        elapsed = time.perf_counter() - self.started
        if elapsed < 0 or (self.samples and elapsed <= self.samples[-1]["t_s"]):
            return
        sample = dict(t_s=elapsed, q_rad=list(joints), component_epoch_s=list(stamps))
        self.writer.writerow([elapsed, feedback_time()] + list(joints)
                             + [math.degrees(x) for x in joints] + list(stamps))
        self.stream.flush()
        self.samples.append(sample)
        self.previous = list(stamps)

    def save(self, status, restored, reason):
        payload = dict(self.metadata, format=FORMAT, robot_model="nero", joint_units="rad",
                       status=status, control_restored=restored, reason=reason, samples=self.samples)
        temporary = self.directory / "trajectory.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "trajectory.json")

    def close(self):
        self.stream.close()


def record_hand(robot, directory, metadata, duration=10.0, hz=20):
    initial = check_before_move(robot)
    if int(robot.get_arm_status().msg.arm_status) == 2:
        raise RuntimeError(
            "이전 NO_SOLUTION이 남아 있습니다. 교시 모드로 전환하지 않았습니다. "
            "python nero_recover.py --execute로 복구를 시도하고(직선 목표로 표시를 지움), "
            "NORMAL 확인 후 다시 기록하세요.")
    _diagnose(robot, "record_before_entry", directory=str(Path(directory).resolve()))
    reported_mode = robot.get_arm_status().msg.mode_feedback
    initial_mode = int(reported_mode)
    if initial_mode not in (0, 1, 2, 3):
        raise RuntimeError(
            "교시 진입 전 이동 모드가 P/J/L/C가 아닙니다: mode_feedback={} (값 {}). "
            "교시 요청/기록은 시작하지 않았습니다. MIT/CPV를 자동 전환하지 않습니다. "
            "팔을 지지하고 disable → prepare --execute → enable 후 상태를 확인하세요.".format(reported_mode, initial_mode))
    initial_pose, _ = flange_pose(robot)
    writer = TeachingFile(directory, dict(metadata, initial_flange_m_rad=initial_pose,
                                          pre_teach_joints_rad=initial,
                                          pre_teach_motion_mode=initial_mode))
    entered = restored = False
    reader = TeachingFeed()
    status, reason, caught = "completed", "", None
    try:
        print("팔과 도구를 지지하세요. 손 유도 모드로 전환합니다. disable은 보내지 않습니다.")
        entered_at = feedback_time()
        entered = True  # Even partial transmission requires explicit cleanup.
        request_teaching_mode(robot)
        _, source, joints, stamps = wait_for_teaching(robot, newer_than=entered_at)
        reader.active = leader_sample if source == "leader_packets" else linkage_sample
        reader.last, reader.history = list(joints), [source]
        writer.metadata["source"] = source
        writer.metadata["mode_confirmation"] = ("fresh leader packets" if source == "leader_packets"
                                                else "fresh normal packets and ctrl_mode=6; not a torque measurement")
        writer.save("recording", False, "")
        print("교시 기록 데이터:", source)
        if joint_error(initial, joints) > math.radians(1):
            raise RuntimeError("교시 모드 진입 중 자세가 1도 넘게 바뀌었습니다. 지지 상태를 확인하세요.")
        writer.started = time.perf_counter()
        writer.append(joints, stamps)
        print("기록 시작: {:.1f}초 동안 공중에서 천천히 움직이세요.".format(duration))
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            joints, stamps = reader(robot)
            writer.append(joints, stamps)
            time.sleep(min(1.0 / hz, max(0, deadline - time.monotonic())))
        print("기록 시간이 끝났습니다. 팔을 멈추고 지지하세요. 정지 확인 구간도 기록합니다.")
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        reason, caught = str(exc) or "사용자 중단", exc
        print("기록을 중단했습니다. 팔을 지지하고 멈추세요. 제어 모드 복귀를 확인합니다.")
    finally:
        if entered:
            try:
                # On failed recording do not extend the damaged file, but still
                # attempt a stationary, feedback-verified exit from hand guidance.
                # No confirmed teaching stream (entry rejected or never seen):
                # let restore_control pick leader or ordinary packets itself.
                final_joints = restore_control(robot, on_sample=writer.append if caught is None else None,
                                               reader=reader if reader.active is not None else None,
                                               motion_mode=initial_mode,
                                               expected_enabled=(True,) * 7)
                restored = True
                if caught is None and writer.samples and joint_error(final_joints, writer.samples[-1]["q_rad"]) > START_TOLERANCE:
                    raise RuntimeError("제어 모드는 복귀했지만 마지막 기록 자세와 0.5도 넘게 달라 기록 완료를 확인하지 못했습니다.")
            except (Exception, KeyboardInterrupt) as exc:
                status = "failed"
                reason = (reason + "; " if reason else "") + "제어 모드 복귀 미확인: " + (str(exc) or "사용자 중단")
                if caught is None:
                    caught = exc
                print("교시 모드가 남아 있을 수 있습니다. 팔을 지지하고 control --execute로 확인하세요.")
        if reader.switches:
            writer.metadata["source"] = reader.source
            writer.metadata["source_switches"] = reader.switches
        try:
            writer.save(status, restored, reason)
        finally:
            writer.close()
    if caught is not None:
        raise RuntimeError("기록 중단: {}. 부분 파일: {}".format(reason, directory)) from caught
    return writer.samples


def load_teaching(path, limits):
    path = Path(path)
    if path.is_dir():
        path = path / "trajectory.json"
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("교시 파일이 10MB보다 큽니다.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("format") != FORMAT or data.get("robot_model") != "nero"
            or data.get("joint_units") != "rad"):
        raise ValueError("지원하는 Nero 교시 파일 형식(rad)이 아닙니다.")
    if data.get("status") != "completed" or data.get("control_restored") is not True:
        raise ValueError("완료·제어 복귀가 확인된 기록만 재생할 수 있습니다. 중단 파일을 자동 재생하지 않습니다.")
    samples = data.get("samples")
    if not isinstance(samples, list) or not 2 <= len(samples) <= 10000:
        raise ValueError("교시 샘플이 2~10000개 필요합니다.")
    previous = None
    for index, sample in enumerate(samples):
        stamp = sample.get("t_s")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("샘플 경과 시간이 유효하지 않습니다.")
        if previous is not None and not 0 < stamp - previous < 0.5:
            raise ValueError("샘플 {}→{}의 시간 간격이 {:.6f}초입니다. "
                             "시간 역전·중복 또는 0.5초 이상 기록 공백이 있습니다.".format(
                                 index - 1, index, stamp - previous))
        validate_joints(sample.get("q_rad", []), limits)
        previous = stamp
    return path, data


def make_waypoints(samples, tolerance_degrees=0.1, step_degrees=2.0, max_span_degrees=360.0):
    """Simplify against time-parametrized joint lines, then bound every step."""
    if not math.isfinite(tolerance_degrees) or not 0.01 <= tolerance_degrees <= 0.5:
        raise ValueError("경로 단순화 허용 오차는 0.01~0.5도입니다.")
    if not math.isfinite(step_degrees) or not 0.2 <= step_degrees <= 4:
        raise ValueError("웨이포인트 간격은 0.2~4도입니다.")
    if not math.isfinite(max_span_degrees) or not 0 < max_span_degrees <= 360:
        raise ValueError("관절별 전체 움직임 범위 한도는 0보다 크고 360도 이하여야 합니다.")
    span = max(max(s["q_rad"][i] for s in samples) - min(s["q_rad"][i] for s in samples) for i in range(7))
    if span > math.radians(max_span_degrees):
        raise ValueError(
            "기록의 관절별 움직임 범위가 {:.3f}도로 소프트웨어 한도 {:.1f}도를 넘습니다. "
            "범위는 기록 중 관절별 최댓값-최솟값입니다. --max-span-deg로 늘리거나 빼세요(최대 360도). "
            "SDK 관절 제한과 웨이포인트 간격 제한은 그대로 적용됩니다.".format(math.degrees(span), max_span_degrees))
    keep = {0, len(samples) - 1}
    stack = [(0, len(samples) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        first, last = samples[lo], samples[hi]
        largest, split = -1.0, None
        for i in range(lo + 1, hi):
            fraction = (samples[i]["t_s"] - first["t_s"]) / (last["t_s"] - first["t_s"])
            line = [a + fraction * (b - a) for a, b in zip(first["q_rad"], last["q_rad"])]
            error = joint_error(samples[i]["q_rad"], line)
            if error > largest:
                largest, split = error, i
        if largest > math.radians(tolerance_degrees):
            keep.add(split)
            stack.extend(((lo, split), (split, hi)))
    # Drop kept points that barely differ from the previous one: each waypoint
    # costs about 0.85 s. On teach_test02 (2026-11-18) this removed 7 of 208
    # waypoints and kept every recorded sample within 0.1 degree of the path.
    kept = sorted(keep)
    selected = [samples[kept[0]]["q_rad"]]
    for index in kept[1:-1]:
        if joint_error(samples[index]["q_rad"], selected[-1]) >= math.radians(MIN_WAYPOINT_DEG):
            selected.append(samples[index]["q_rad"])
    selected.append(samples[kept[-1]]["q_rad"])
    points = [list(selected[0])]
    for target in selected[1:]:
        start = points[-1]
        count = max(1, math.ceil(joint_error(start, target) / math.radians(step_degrees)))
        points.extend([[a + (b - a) * i / count for a, b in zip(start, target)] for i in range(1, count + 1)])
        if len(points) > 1000:
            raise ValueError("웨이포인트가 1000개를 넘습니다. 더 짧은 동작으로 기록하세요.")
    return points, math.degrees(span)


def move_to_start(robot, start, *, speed_percent=5, timeout=30.0, min_z_mm=50.0, start_delay=0.0, check=None):
    """One joint move straight to the recording's first posture; all joints arrive together.

    The default way back since 2026-11-18. Retracing the recording took about
    0.85 s per waypoint (teach_test02: 201 waypoints, about 3 minutes) and could
    stall on a single unapplied frame; this is one move of a few seconds.
    The path differs from the recorded one, so the arm sweeps through space the
    hand did not visit: it is only checked for joint limits and for elbow,
    wrist and flange staying above ``min_z_mm``. No collision check.
    """
    current = motion_feedback(robot)[0]
    travel = math.degrees(joint_error(current, start))
    path_mm, lowest_mm = joint_move_trace(current, start, together=True)
    floor_mm = height_floor_mm(current, min_z_mm)
    limit = height_limit_text(floor_mm, min_z_mm)
    print("시작 자세로 바로 이동: 가장 많이 도는 관절 {:.1f}도 / 플랜지 이동 약 {:.0f}mm / "
          "팔꿈치·손목·플랜지 최저 높이 {:.0f}mm ({})".format(travel, path_mm, lowest_mm, limit))
    if lowest_mm < floor_mm:
        raise RuntimeError("시작 자세로 바로 가면 높이 {:.0f}mm까지 내려갑니다 ({}). "
                           "--return-path recorded로 기록을 거꾸로 따라가세요.".format(lowest_mm, limit))
    if travel > 90:
        print("큰 관절 이동입니다. 팔 전체가 넓게 휘돌아 가므로 주변 공간을 확인하세요.")
    if start_delay > 0:
        print("{:g}초 후 이동을 시작합니다. 중단하려면 Ctrl+C.".format(start_delay))
        time.sleep(start_delay)
    return move_joint_target(robot, start, timeout=timeout, max_step_degrees=360.0, allow_long_move=True,
                             lead_degrees=max(2.0, 0.3 * speed_percent), speed_percent=speed_percent,
                             on_feedback=check)


def execute_path(robot, points, *, reverse=False, cycles=1, return_to_start=False,
                 timeout=30.0, step_degrees=2.0, output_dir="data/nero_teach/runs", source="",
                 return_path="direct", speed_percent=5, min_z_mm=50.0, start_delay=0.0,
                 replay_mode="smooth"):
    """Replay ``points`` (or, with ``reverse``, return to their start).

    ``return_path``: "direct" goes back to the first posture with one joint
    move (move_to_start); "recorded" retraces the recording waypoint by
    waypoint, a path the hand already showed to be clear. It applies to the
    home command, the return between cycles and the final return.
    """
    if return_path not in ("direct", "recorded"):
        raise ValueError("복귀 방식은 direct 또는 recorded입니다.")
    if replay_mode not in ("smooth", "waypoints"):
        raise ValueError("재생 방식은 smooth 또는 waypoints입니다.")
    current = check_before_move(robot)
    route = list(reversed(points)) if reverse else list(points)
    if reverse and joint_error(current, points[0]) <= math.radians(0.1):
        print("이미 기록의 시작 자세에 있습니다. 복귀 이동을 보내지 않습니다.")
        return None
    direct_home = reverse and return_path == "direct"
    if not direct_home and joint_error(current, route[0]) > START_TOLERANCE:
        raise RuntimeError("현재 자세가 기록의 {} 자세에서 0.5도 넘게 벗어났습니다. 임의 연결 이동은 하지 않습니다.{}".format(
            "끝" if reverse else "시작",
            " 기본 복귀 방식(direct)은 어느 자세에서든 시작 자세로 갑니다." if reverse else
            " 먼저 python nero_teach.py home <기록> --execute로 시작 자세로 돌아가세요."))
    way_back = None if return_path == "direct" else list(reversed(points))  # None: one direct joint move.
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    recorder = RunRecorder(robot, Path(output_dir) / run_id,
                           dict(schema_version=1, created_at=datetime.now().astimezone().isoformat(),
                                mode="teach_home" if reverse else "teach_replay", source=str(source),
                                coordinate_frame="robot_base", control_point="flange",
                                units="CSV: mm, deg, s; waypoints: rad",
                                waypoints_rad=route, speed_percent=speed_percent, return_path=return_path,
                                replay_mode=replay_mode,
                                notes="Joint path replay; original timing and contact force are not reproduced."),
                           enforce_motion_state=True)
    result, reason = "completed", ""
    try:
        recorder.start()
        observe(recorder, 0.2)
        for cycle in range(1, cycles + 1):
            legs = [("home", way_back if reverse else None)] if reverse else [("replay", route)]
            if cycle > 1:
                legs.insert(0, ("cycle_return", way_back))
            if return_to_start and cycle == cycles and not reverse:
                legs.append(("final_return", way_back))
            for phase, leg in legs:
                if leg is None:  # Direct return: one joint move to the first posture.
                    recorder.set_phase(cycle, phase)
                    recorder.event("waypoint_requested", index=0, target_rad=points[0], direct=True)
                    print("{} (시작 자세로 바로 이동) / 실행 {}/{}".format(phase, cycle, cycles))
                    actual = move_to_start(robot, points[0], speed_percent=speed_percent, timeout=timeout,
                                           min_z_mm=min_z_mm, check=recorder.check,
                                           start_delay=start_delay if phase == "home" else 0.0)
                    recorder.segments.append(dict(cycle=cycle, phase=phase, index=0, direct=True,
                        target_rad=points[0], actual_rad=actual, endpoint_error_mm=None,
                        max_joint_error_deg=math.degrees(joint_error(actual, points[0]))))
                    recorder.event("waypoint_completed", **recorder.segments[-1])
                    continue
                if replay_mode == "smooth":
                    recorder.set_phase(cycle, phase)
                    recorder.event("path_requested", points=len(leg))
                    print("{} (연속 재생, 점 {}개) / 실행 {}/{}".format(phase, len(leg), cycle, cycles))
                    actual, deviation = follow_joint_path(
                        robot, leg, timeout=timeout, speed_percent=speed_percent,
                        lead_degrees=max(2.0, 0.3 * speed_percent), on_feedback=recorder.check)
                    recorder.segments.append(dict(cycle=cycle, phase=phase, index=len(leg) - 1, smooth=True,
                        target_rad=leg[-1], actual_rad=actual, endpoint_error_mm=None,
                        max_path_deviation_deg=deviation,
                        max_joint_error_deg=math.degrees(joint_error(actual, leg[-1]))))
                    recorder.event("path_completed", **recorder.segments[-1])
                    continue
                for index, target in enumerate(leg):
                    recorder.set_phase(cycle, phase)
                    recorder.event("waypoint_requested", index=index, target_rad=target)
                    print("{} {}/{} / 실행 {}/{}".format(phase, index + 1, len(leg), cycle, cycles))
                    actual = move_joint_target(robot, target, timeout=timeout,
                                               max_step_degrees=step_degrees + 0.5,
                                               on_feedback=recorder.check, speed_percent=speed_percent)
                    recorder.segments.append(dict(cycle=cycle, phase=phase, index=index,
                        target_rad=target, actual_rad=actual, endpoint_error_mm=None,
                        max_joint_error_deg=math.degrees(joint_error(actual, target))))
                    recorder.event("waypoint_completed", **recorder.segments[-1])
        observe(recorder, 0.2)
    except (Exception, KeyboardInterrupt) as exc:
        result, reason = "failed", str(exc) or "사용자 중단"
        raise
    finally:
        summary = recorder.finish(result, reason)
        print("재생/복귀 기록:", recorder.directory.resolve())
        if result == "completed" and summary["status"] != "completed":
            raise RuntimeError("재생 기록을 정상 완료하지 못했습니다: {}".format(summary["reason"]))
    return recorder.directory


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    record = commands.add_parser("record", help="손 유도 관절 경로 기록")
    record.add_argument("--output", help="새 기록 폴더. 기존 폴더는 덮어쓰지 않음")
    record.add_argument("--duration", type=float, default=10, help="기록 시간(초), 1~120")
    record.add_argument("--sample-hz", type=float, default=20, help="기록 폴링 목표 Hz, 5~50")
    record.add_argument("--return-to-start", action="store_true", help="기록/제어 복귀 후 시작 자세로 복귀")
    for name in ("inspect", "home", "replay"):
        sub = commands.add_parser(name, help={"inspect": "파일/경로 검토 (오프라인)",
            "home": "기록의 시작 자세로 복귀 (기본: 관절 이동 한 번)", "replay": "기록한 관절 경로 재생"}[name])
        sub.add_argument("path", help="기록 폴더 또는 trajectory.json")
    control = commands.add_parser("control", help="관측 재개 및 정지 확인 후 제어 모드로 복귀")
    control.add_argument("--firmware", choices=("v112", "v120", "v121"),
                         help="확인된 펌웨어를 명시해 무응답 상태에서 조회 생략. 1.12는 v112")
    commands.add_parser("prepare", help="모터 7개 비활성 상태에서 일반 MOVE_J 모드 준비. 자동 enable/disable 없음")
    for sub in commands.choices.values():
        sub.add_argument("--execute", action="store_true", help="실제 모드 전환/이동. 생략하면 설명/미리보기만")
        sub.add_argument("--channel", default="0")
        sub.add_argument("--simplify-deg", type=float, default=0.1)
        sub.add_argument("--step-deg", type=float, default=2.0)
        sub.add_argument("--max-span-deg", type=float, default=360.0,
                         help="기록 중 관절별 움직임 범위 한도(도), 0~360. 기본 360(사실상 제한 없음)")
        sub.add_argument("--timeout", type=float, default=30)
        sub.add_argument("--return-path", choices=("direct", "recorded"), default="direct",
                         help="시작 자세로 돌아가는 방식. direct(기본): 관절 이동 한 번 / "
                              "recorded: 기록을 거꾸로 따라감(손으로 지나간 길, 느림)")
        sub.add_argument("--speed", type=int, default=DEFAULT_SPEED_PERCENT,
                         help="속도 비율(%%), 1~{}. 기본 {}".format(MAX_SPEED_PERCENT, DEFAULT_SPEED_PERCENT))
        sub.add_argument("--min-z-mm", type=float, default=50.0,
                         help="direct 복귀 중 팔꿈치·손목·플랜지 최저 허용 높이(mm). 기본 50. 충돌 검사는 아님")
        sub.add_argument("--replay-mode", choices=("smooth", "waypoints"), default="smooth",
                         help="smooth(기본): 경로를 멈추지 않고 연속으로 따라감 / "
                              "waypoints: 2도 간격 점마다 멈췄다 가는 이전 방식")
    replay = commands.choices["replay"]
    replay.add_argument("--cycles", type=int, default=1, help="재생 횟수, 1~10. 사이에는 시작 자세로 복귀")
    replay.add_argument("--return-to-start", action="store_true", help="마지막 재생 후에도 시작 자세로 복귀")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    robot = None
    diagnostics = None
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout은 0보다 큰 유한한 초 값이어야 합니다.")
        if not 1 <= getattr(args, "cycles", 1) <= 10:
            raise ValueError("재생 횟수는 1~10입니다.")
        if not 1 <= args.speed <= MAX_SPEED_PERCENT:
            raise ValueError("속도 비율은 1~{}%입니다.".format(MAX_SPEED_PERCENT))
        if not math.isfinite(args.min_z_mm):
            raise ValueError("최저 높이는 유한한 mm 값이어야 합니다.")
        make_waypoints([dict(t_s=0, q_rad=[0.0] * 7), dict(t_s=0.1, q_rad=[0.0] * 7)],
                       args.simplify_deg, args.step_deg, args.max_span_deg)
        if args.command == "record":
            if not math.isfinite(args.duration) or not 1 <= args.duration <= 120:
                raise ValueError("기록 시간은 1~120초입니다.")
            if not math.isfinite(args.sample_hz) or not 5 <= args.sample_hz <= 50:
                raise ValueError("기록 폴링은 5~50Hz입니다.")
        if args.command in ("record", "control", "prepare") and not args.execute:
            print("미리보기: 실제 {}에는 --execute가 필요합니다. 연결/모드 전환/이동하지 않았습니다.".format(args.command))
            print("record: 이미 활성화된 팔을 손 유도 모드로 전환해 기록 후 정지 확인 및 제어 복귀.")
            print("control: 손 유도 상태의 팔을 지지하고 멈춘 뒤 제어 모드로 복귀.")
            print("prepare: 모터 7개 비활성 상태에서만 MOVE_J로 변경. 목표/enable/disable/reset 전송 없음.")
            return 0
        limits = create_agx_arm_config(robot=ArmModel.NERO)["joint_limits"]
        if args.command in ("inspect", "home", "replay"):
            path, data = load_teaching(args.path, limits)
            points, span = make_waypoints(data["samples"], args.simplify_deg, args.step_deg, args.max_span_deg)
            print("원본 {}개 → 재생 {}개 / 최대 관절별 범위 {:.3f}도".format(len(data["samples"]), len(points), span))
            print("기록 시간 {:.3f}초. 재생은 {}% 속도의 관절 경로 근사이며 원래 시간/힘을 재현하지 않습니다.".format(
                data["samples"][-1]["t_s"] - data["samples"][0]["t_s"], args.speed))
            if args.command in ("home", "replay"):
                print("시작 자세로 돌아가는 방식: {}".format(
                    "관절 이동 한 번 (direct, 거리·최저 높이는 실행 시 현재 자세에서 계산)"
                    if args.return_path == "direct" else "기록을 거꾸로 따라감 (recorded)"))
            print("시작 관절각(deg):", [round(math.degrees(x), 3) for x in points[0]])
            print("끝 관절각(deg):", [round(math.degrees(x), 3) for x in points[-1]])
            if args.command == "inspect" or not args.execute:
                print("파일 검토만 완료. 로봇 연결/모드 전환/이동은 하지 않았습니다. 충돌 검사는 수행하지 않습니다.")
                return 0
        diagnostics = SessionDiagnostics()
        print("실행 진단 로그:", diagnostics.path.resolve())
        diagnostics.event(None, "start", command=args.command, options=vars(args))
        if args.command == "control" and args.firmware is not None:
            robot, firmware, profile = connect_known_nero(args.channel, args.firmware)
        else:
            robot, firmware, profile = connect_nero(args.channel)
        robot._nero_teach_diagnostics = diagnostics
        if str(profile) not in ("v112", "v120", "v121"):
            raise RuntimeError("이 도구는 Nero 펌웨어 1.12 이상의 leader 패킷 형식을 사용합니다.")
        print("펌웨어: {} / SDK: {}. 자동 enable/disable/reset은 하지 않습니다.".format(firmware, profile))
        time.sleep(1)
        _diagnose(robot, "connected", firmware=firmware, sdk_profile=str(profile))
        if args.command == "record":
            directory = args.output or str(Path("data/nero_teach") / (
                datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]))
            if Path(directory).exists():
                raise ValueError("기록 폴더가 이미 있습니다. 새 --output 이름을 사용하세요.")
            metadata = dict(created_at=datetime.now().astimezone().isoformat(), firmware=firmware,
                            sdk_profile=str(profile), sample_hz=args.sample_hz,
                            diagnostic_log=str(diagnostics.path.resolve()),
                            source="unconfirmed", entry_method="role_only_preserve_normal_feedback",
                            mode_confirmation="unconfirmed")
            print("교시 저장 폴더:", Path(directory).resolve())
            print("3초 후 준비 상태를 확인합니다. 팔과 도구를 지지하고 천천히 움직일 준비를 하세요.")
            time.sleep(3)
            record_hand(robot, directory, metadata, args.duration, args.sample_hz)
            print("기록 완료. 제어 모드 복귀를 확인했습니다.")
            if args.return_to_start:
                path, data = load_teaching(directory, robot.get_config()["joint_limits"])
                points, _ = make_waypoints(data["samples"], args.simplify_deg, args.step_deg, args.max_span_deg)
                print("3초 후 시작 자세로 복귀합니다({}). 팔에서 손을 떼고 이동 공간을 비우세요.".format(args.return_path))
                time.sleep(3)
                execute_path(robot, points, reverse=True, timeout=args.timeout,
                             step_degrees=args.step_deg, source=path, return_path=args.return_path,
                             speed_percent=args.speed, min_z_mm=args.min_z_mm, replay_mode=args.replay_mode)
        elif args.command == "control":
            # Normal telemetry alone does not prove the role is follower. An
            # explicit control request observes a stop then requests follower.
            restore_control(robot)
        elif args.command == "prepare":
            prepare_joint_mode(robot, timeout=args.timeout)
        else:
            # Revalidate against the live SDK limits as well as offline defaults.
            load_teaching(args.path, robot.get_config()["joint_limits"])
            execute_path(robot, points, reverse=args.command == "home",
                         cycles=getattr(args, "cycles", 1),
                         return_to_start=getattr(args, "return_to_start", False),
                         timeout=args.timeout, step_degrees=args.step_deg, source=path,
                         return_path=args.return_path, speed_percent=args.speed, min_z_mm=args.min_z_mm,
                         start_delay=START_DELAY_S if args.command == "home" else 0.0,
                         replay_mode=args.replay_mode)
        _diagnose(robot, "completed")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if diagnostics is not None:
            diagnostics.event(robot, "failed", error=str(exc) or "사용자 중단")
        offline_review = (args.command == "inspect"
                          or (args.command in ("home", "replay") and not args.execute))
        if offline_review:
            print("파일 검토 중단:", str(exc) or "사용자 중단")
            print("오프라인 검토만 수행했습니다. 로봇 연결/모드 전환/이동은 하지 않았습니다.")
        else:
            print("교시 도구 중단:", str(exc) or "사용자 중단")
            print("자동 재시도/임의 복귀/reset은 하지 않습니다. 현재 상태를 확인하세요.")
        if args.command == "control" and args.firmware is None:
            print("펌웨어 조회도 무응답이면 이미 확인한 버전을 --firmware로 지정할 수 있습니다. 1.12는 v112입니다.")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            if diagnostics is not None:
                diagnostics.close()


if __name__ == "__main__":
    raise SystemExit(main())
