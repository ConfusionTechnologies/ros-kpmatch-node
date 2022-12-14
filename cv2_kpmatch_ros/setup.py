from glob import glob

from setuptools import setup

package_name = "cv2_kpmatch_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="John-Henry Lim",
    maintainer_email="42513874+Interpause@users.noreply.github.com",
    description="image keypoint matching using openCV",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["kpdetector = cv2_kpmatch_ros.code:main"]},
)
