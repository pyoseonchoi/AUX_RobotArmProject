"""Recovery acknowledgement and failure containment on virtual CAN, no hardware."""

import math
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from nero_move_joint import RejectionWatch
from nero_recover import main, recover_no_solution
from nero_teach import record_hand
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_stop_policy import JOINT_IDS, PolicySlave


class RecoveryTests(VirtualRobotCase):
    slave_type = PolicySlave

    def leave_no_solution(self):
        self.device.reported_arm_status = 2
        time.sleep(0.08)

    def wait_for_stop(self):
        deadline = time.monotonic() + 1
        while not any(f.arbitration_id == 0x150 and f.data[0] == 1
                      for f in self.device.host_frames):
            if time.monotonic() >= deadline:
                self.fail("No stop frame")
            time.sleep(0.01)

    def test_current_joint_target_and_verified_normal_without_power_commands(self):
        self.leave_no_solution()
        target = list(self.robot.get_joint_angles().msg)
        recovered = recover_no_solution(self.robot)
        self.assertLess(max(abs(a - b) for a, b in zip(target, recovered)), math.radians(0.002))
        self.assertEqual(int(self.robot.get_arm_status().msg.arm_status), 0)
        frames = self.device.host_frames
        self.assertTrue(all(f.arbitration_id in (0x151,) + JOINT_IDS for f in frames))
        self.assertEqual([f.arbitration_id for f in frames if f.arbitration_id in JOINT_IDS], list(JOINT_IDS))
        encoded = []
        for frame in frames:
            if frame.arbitration_id in JOINT_IDS:
                encoded.append(int.from_bytes(frame.data[:4], "big", signed=True))
                if frame.arbitration_id != 0x170:
                    encoded.append(int.from_bytes(frame.data[4:], "big", signed=True))
        for value, expected in zip(encoded, target):
            self.assertAlmostEqual(value / 1000, math.degrees(expected), delta=0.002)

    def test_normal_is_noop(self):
        recover_no_solution(self.robot)
        self.assertFalse(self.device.host_frames)

    def test_persistent_error_is_not_reported_as_success_even_if_target_reached(self):
        self.leave_no_solution()
        self.device.clear_on_target = False
        with patch.object(RejectionWatch, "STALL_S", 0.25):
            with self.assertRaisesRegex(RuntimeError, "남아 있고"):
                recover_no_solution(self.robot)
        self.assertFalse(any(f.arbitration_id == 0x150 for f in self.device.host_frames))

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

    def test_drift_after_target_requests_stop(self):
        self.leave_no_solution()
        original = self.robot.move_j

        def drift(target):
            original(target)
            time.sleep(0.06)
            self.device.current_joints[0] += math.radians(0.7)
            time.sleep(0.06)

        with patch.object(self.robot, "move_j", side_effect=drift):
            with self.assertRaisesRegex(RuntimeError, "0.5도"):
                recover_no_solution(self.robot)
        self.wait_for_stop()

    def test_missing_joint_component_after_target_never_confirms_recovery(self):
        self.leave_no_solution()
        original_move = self.robot.move_j
        original_send = self.device._send_and_record
        blocked = set()

        def send(frame):
            if frame.arbitration_id not in blocked:
                original_send(frame)

        def move(target):
            blocked.add(0x2A6)  # J3/J4: others keep updating.
            original_move(target)

        with patch.object(self.device, "_send_and_record", side_effect=send), \
             patch.object(self.robot, "move_j", side_effect=move):
            with self.assertRaisesRegex(RuntimeError, "피드백"):
                recover_no_solution(self.robot)
        self.wait_for_stop()

    def test_interrupt_during_target_requests_stop(self):
        self.leave_no_solution()
        with patch.object(self.robot, "move_j", side_effect=KeyboardInterrupt):
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
