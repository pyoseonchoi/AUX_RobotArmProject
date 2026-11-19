"""NO_SOLUTION recovery with straight-line targets, on virtual CAN only."""

from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from nero_move_joint import RejectionWatch
from nero_recover import main, recover_no_solution
from nero_teach import record_hand
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_stop_policy import JOINT_IDS, POSE_IDS, PolicySlave


class FirmwareFlagSlave(PolicySlave):
    """Like firmware 1.12: joint targets keep NO_SOLUTION, a pose target clears it.

    ``ignored_pose_targets`` complete pose targets keep the flag before one clears it.
    """

    def __init__(self, channel):
        self.ignored_pose_targets = 0
        self.pose_targets = 0
        super().__init__(channel)

    def _apply_host_command(self, aid, data):
        flag = self.reported_arm_status
        if aid == POSE_IDS[0]:
            self.pose_targets += 1
        super()._apply_host_command(aid, data)
        if aid in JOINT_IDS or (aid in POSE_IDS and self.pose_targets <= self.ignored_pose_targets):
            self.reported_arm_status = flag


def z_targets_mm(frames):
    return [int.from_bytes(f.data[:4], "big", signed=True) / 1000 for f in frames if f.arbitration_id == 0x153]


class RecoveryTests(VirtualRobotCase):
    slave_type = FirmwareFlagSlave

    def leave_no_solution(self):
        self.device.reported_arm_status = 2
        time.sleep(0.08)

    def wait_for_stop(self):
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1 for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No stop frame")
            time.sleep(0.01)

    def status(self):
        return int(self.robot.get_arm_status().msg.arm_status)

    def test_pose_target_at_the_current_position_clears_the_flag(self):
        self.leave_no_solution()
        z = self.robot.get_flange_pose().msg[2] * 1000
        recover_no_solution(self.robot)
        self.assertEqual(self.status(), 0)
        frames = self.device.host_frames
        self.assertFalse(any(f.arbitration_id in JOINT_IDS + (0x150, 0x471) for f in frames))
        self.assertEqual(z_targets_mm(frames), [round(z, 3)])  # One target: where the flange already is.

    def test_flag_that_survives_the_first_try_is_cleared_by_a_1mm_move(self):
        self.leave_no_solution()
        self.device.ignored_pose_targets = 1
        z = self.robot.get_flange_pose().msg[2] * 1000
        with patch.object(RejectionWatch, "STALL_S", 0.3):
            recover_no_solution(self.robot)
        self.assertEqual(self.status(), 0)
        self.assertEqual(z_targets_mm(self.device.host_frames), [round(z, 3), round(z - 1, 3), round(z, 3)])
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))

    def test_flag_that_never_clears_fails_without_stop_and_ends_in_place(self):
        self.leave_no_solution()
        self.device.ignored_pose_targets = 99
        z = self.robot.get_flange_pose().msg[2] * 1000
        with patch.object(RejectionWatch, "STALL_S", 0.3):
            with self.assertRaisesRegex(RuntimeError, "지워지지 않았습니다"):
                recover_no_solution(self.robot)
        self.assertEqual(self.status(), 2)
        self.assertEqual(z_targets_mm(self.device.host_frames)[-1], round(z, 3))
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))

    def test_joint_target_does_not_clear_the_flag(self):
        # The reason this script no longer sends joint targets (2026-11-17 18:29).
        self.leave_no_solution()
        self.robot.move_j(list(self.robot.get_joint_angles().msg))
        time.sleep(0.1)
        self.assertEqual(self.status(), 2)

    def test_normal_is_noop(self):
        recover_no_solution(self.robot)
        self.assertFalse(self.device.host_frames)

    def test_stop_disabled_and_mit_states_send_no_commands(self):
        for state in ("stop", "disabled", "mit"):
            with self.subTest(state=state):
                self.device.reported_arm_status = 1 if state == "stop" else 2
                self.device._joints_enabled = [state != "disabled"] * 7
                self.device._mode_feedback = 6 if state == "mit" else 1
                time.sleep(0.08)
                with self.assertRaises(RuntimeError):
                    recover_no_solution(self.robot)
                self.assertFalse(self.device.host_frames)

    def test_feedback_lost_after_a_target_is_reported_and_nothing_more_is_sent(self):
        self.leave_no_solution()
        self.device.ignored_pose_targets = 1
        original_move = self.robot.move_l
        original_send = self.device._send_and_record
        blocked = set()

        def send(frame):
            if frame.arbitration_id not in blocked:
                original_send(frame)

        def move(target):
            blocked.add(0x2A6)  # J3/J4 stop arriving; the others keep updating.
            original_move(target)

        with patch.object(self.device, "_send_and_record", side_effect=send), \
             patch.object(self.robot, "move_l", side_effect=move):
            with self.assertRaisesRegex(RuntimeError, "피드백"):
                recover_no_solution(self.robot)
        self.assertEqual(len(z_targets_mm(self.device.host_frames)), 1)  # No 1 mm move without feedback.

    def test_interrupt_during_target_requests_stop(self):
        self.leave_no_solution()
        with patch.object(self.robot, "move_l", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                recover_no_solution(self.robot)
        self.wait_for_stop()

    def test_default_cli_is_read_only(self):
        self.leave_no_solution()
        with patch("nero_recover.connect_nero", return_value=(self.robot, "1.12", "v112")), \
             patch("sys.argv", ["nero_recover.py"]):
            self.assertEqual(main(), 0)
        self.assertFalse(self.device.host_frames)

    def test_teaching_rejects_before_mode_switch_and_file_creation(self):
        self.leave_no_solution()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record"
            with self.assertRaisesRegex(RuntimeError, "nero_recover.py"):
                record_hand(self.robot, output, {}, duration=1)
            self.assertFalse(output.exists())
        self.assertFalse(self.device.host_frames)
