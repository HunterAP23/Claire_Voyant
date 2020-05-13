import tensorflow as tf

def enable_eager_exec():
    print("Enabling Eager Execution.")
    tf.compat.v1.enable_eager_execution()
    print("Eager Execution is enabled.")


def disable_eager_exec():
    print("Disabling Eager Execution.")
    tf.python.framework.ops.disable_eager_execution()
    print("Eager Execution is disabled.")


def check_eager_exec():
    if(tf.executing_eagerly()):
        print("Eager Execution is enabled.")
    else:
        print("Eager Execution is disabled.")


def check_devices():
    print(("Is your GPU available for use?\n{0}").format(
    "Yes, your GPU is available: True" if tf.test.is_gpu_available() == True else "No, your GPU is NOT available: False"
))

    print(("\nYour devices that are available:\n{0}").format(
        [device.name for device in tf.config.experimental.list_physical_devices()]
    ))



check_eager_exec()
# check_devices()
