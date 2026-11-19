"""Local command diagnostics. Reading a snapshot never sends robot commands."""

from datetime import datetime
import json
from pathlib import Path
import time
import uuid

from nero_read_status import feedback_time, recent

DEFAULT_ROOT = Path("data/nero_command_logs")

class SessionDiagnostics:
    def __init__(self, root=None):
        root = Path(DEFAULT_ROOT if root is None else root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / (datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_")
                            + uuid.uuid4().hex[:8] + ".jsonl")
        self.stream = self.path.open("x", encoding="utf-8")
        self.started = time.monotonic()

    def event(self, robot, event, **values):
        row = dict(event=event, time=datetime.now().astimezone().isoformat(),
                   elapsed_s=time.monotonic() - self.started, **values)
        if robot is not None:
            try:
                status = robot.get_arm_status()
                row["status_fresh"] = recent(status)
                if status is not None:
                    row["status_age_s"] = feedback_time() - status.timestamp
                    for name in ("ctrl_mode", "arm_status", "mode_feedback", "motion_status"):
                        value = getattr(status.msg, name)
                        row[name], row[name + "_value"] = str(value), int(value)
                drivers = [robot.get_driver_states(i) for i in range(1, 8)]
                row["driver_feedback_fresh"] = [recent(d) for d in drivers]
                row["enabled"] = [None if d is None else bool(d.msg.foc_status.driver_enable_status)
                                  for d in drivers]
                for key, feedback in (("joints_rad", robot.get_joint_angles()),
                                      ("flange_m_rad", robot.get_flange_pose())):
                    row[key] = None if feedback is None else list(feedback.msg)
                    row[key + "_fresh"] = recent(feedback)
                row["comm_error"] = str(robot.get_comm_error()) if robot.get_comm_error() else None
                row["interface"] = robot.get_config().get("interface")
            except Exception as exc:
                row["snapshot_error"] = str(exc)
        try:
            self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.stream.flush()
        except (OSError, ValueError) as exc:
            # Never let a diagnostic write prevent emergency stopping/cleanup.
            print("진단 로그 저장 실패:", exc)

    def close(self):
        self.stream.close()


def diagnose(robot, event, **values):
    logger = getattr(robot, "_nero_diagnostics", None)
    if logger is not None:
        logger.event(robot, event, **values)
