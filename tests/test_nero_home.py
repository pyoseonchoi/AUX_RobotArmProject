"""Home capture and return use virtual CAN only; never connect to hardware."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from nero_home import capture_home, load_home, main, return_home
from nero_read_status import print_status
from tests.test_nero_move_to import VirtualRobotCase
from tests.test_nero_stop_policy import PolicySlave, JOINT_IDS


class HomeTests(VirtualRobotCase):
    slave_type = PolicySlave

    def home(self):
        return capture_home(self.robot, "1.12", "v112")

    def test_capture_with_disabled_motors_is_read_only(self):
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        home = self.home()
        self.assertEqual(home["joints_rad"], list(self.robot.get_joint_angles().msg))
        self.assertEqual(home["flange_pose_m_rad"], list(self.robot.get_flange_pose().msg))
        self.assertFalse(self.device.host_frames)

    def test_partial_pose_feedback_cannot_be_displayed_or_saved(self):
        original = self.device._send_and_record
        def send(frame):
            if frame.arbitration_id != 0x2A2:
                original(frame)
        with patch.object(self.device, "_send_and_record", side_effect=send):
            time.sleep(0.6)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_status(self.robot)
            self.assertIn("현재 XYZ 확인 불가", output.getvalue())
            self.assertNotIn("X=", output.getvalue())
            with self.assertRaisesRegex(RuntimeError, "피드백"):
                capture_home(self.robot, "1.12", "v112", timeout=0.2)
        self.assertFalse(self.device.host_frames)

    def test_moving_arm_is_not_saved(self):
        from nero_home import read_posture
        calls = [0]
        def changing(*args, **kwargs):
            angles, pose = read_posture(*args, **kwargs)
            calls[0] += 1
            angles[0] += math.radians(calls[0])
            return angles, pose
        with patch("nero_home.read_posture", side_effect=changing):
            with self.assertRaisesRegex(RuntimeError, "자세가 변하고"):
                self.home()
        self.assertFalse(self.device.host_frames)

    def test_return_sends_saved_joint_target_without_power_commands(self):
        home = self.home()
        home["joints_rad"][0] += math.radians(3)
        return_home(self.robot, home, timeout=2)
        self.assertAlmostEqual(self.robot.get_joint_angles().msg[0], home["joints_rad"][0], places=4)
        frames = self.device.host_frames
        self.assertTrue(any(f.arbitration_id in JOINT_IDS for f in frames))
        self.assertTrue(all(f.arbitration_id in (0x151,) + JOINT_IDS for f in frames))

    def test_already_home_sends_no_target(self):
        return_home(self.robot, self.home())
        self.assertFalse(self.device.host_frames)

    def test_disabled_motors_and_excess_travel_block_return(self):
        home = self.home()
        home["joints_rad"][0] += math.radians(3)
        with self.assertRaisesRegex(RuntimeError, "허용 간격"):
            return_home(self.robot, home, max_joint_move_deg=1)
        self.device._joints_enabled = [False] * 7
        time.sleep(0.08)
        with self.assertRaisesRegex(RuntimeError, "활성화"):
            return_home(self.robot, home)
        self.assertFalse(self.device.host_frames)

    def test_flange_mismatch_is_not_success(self):
        home = self.home()
        home["joints_rad"][0] += math.radians(1)
        home["flange_pose_m_rad"][0] += 0.01
        with self.assertRaisesRegex(RuntimeError, "저장한 플랜지 자세와 다릅니다"):
            return_home(self.robot, home)

    def test_preview_and_overwrite_guard(self):
        home = self.home()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "home.json"
            path.write_text(json.dumps(home), encoding="utf-8")
            before = path.read_bytes()
            with patch("nero_home.connect_nero", return_value=(self.robot, "1.12", "v112")), \
                 patch("sys.argv", ["nero_home.py", "--position", str(path)]):
                self.assertEqual(main(), 0)
            self.assertFalse(self.device.host_frames)
            with patch("nero_home.connect_nero") as connect, \
                 patch("sys.argv", ["nero_home.py", "--position", str(path), "--save-current"]):
                self.assertEqual(main(), 1)
                connect.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            home["joints_rad"][0] = float("nan")
            path.write_text(json.dumps(home), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_home(path)
