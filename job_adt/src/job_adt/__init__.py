import os

__version__ = "0.1.0"

# IMPORTANT: Prevent submitit package from silencing Dataflow logs!
#
# demucs depends on "dora", which in turn depends on "submitit". Importing
# submitit has an incompatibility with GCP Dataflow. To prevent submitit from
# interfering with logging, set this environment variable. See:
# https://github.com/facebookincubator/submitit/blob/4cf1462d7216f9dcc530daeb703ce07c37cf9d72/submitit/core/logger.py#L11-L16
os.environ["SUBMITIT_LOG_LEVEL"] = "NOCONFIG"