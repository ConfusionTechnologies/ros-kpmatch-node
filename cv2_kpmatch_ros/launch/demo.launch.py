from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    kpdet_node = Node(
        package="cv2_kpmatch_ros",
        namespace="/",
        executable="kpdetector",
        name="kpdet",
        parameters=[{"frames_in_topic": "/rtc/rtc_receiver/frames_out"}],
        respawn=True,
    )

    aiortc_cfg = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("aiortc_ros")) / "main.launch.py")
        ),
        launch_arguments=[("namespace", "/rtc")],
    )

    return LaunchDescription([kpdet_node, aiortc_cfg])

