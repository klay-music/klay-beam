import setuptools

# DO NOT DELETE THIS FILE!
#
# Even though it is just a stub, it is required to correctly package the
# klay_beam module for running on Dataflow. We use the dataflow --setup_file
# option to specify this file when launching a Dataflow job. This is how we
# instruct (locally running) python code to create an sdist package and upload
# it to GCS. The sdist package is then downloaded by the workers.

if __name__ == "__main__":
    setuptools.setup()
