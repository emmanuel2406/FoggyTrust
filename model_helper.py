from __future__ import print_function

from mxnet import gluon


def get_cnn(num_outputs=10):
    """Legacy CNN used for MNIST/FashionMNIST experiments."""
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation="relu"))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation="relu"))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(int(num_outputs)))
    return cnn


def _load_gluoncv_resnet20():
    try:
        from gluoncv.model_zoo import get_model
    except ImportError as exc:
        raise ImportError(
            "ResNet-20 model selection requires GluonCV. "
            "Install it with `pip install gluoncv`."
        ) from exc
    return get_model


def get_resnet20(num_outputs=10, pretrained=False):
    """
    CIFAR-style ResNet-20 from GluonCV model zoo.

    Uses `cifar_resnet20_v1` and resets classifier width to `num_outputs`.
    """
    get_model = _load_gluoncv_resnet20()
    net = get_model("cifar_resnet20_v1", classes=int(num_outputs), pretrained=bool(pretrained))
    return net


def resolve_model_type(dataset, requested_net):
    """
    Enforce dataset-driven model policy.

    - MNIST/FashionMNIST: CNN
    - Everything else (including SnapshotSafari): ResNet-20
    """
    dataset_key = str(dataset).strip().lower()
    if dataset_key in ("fashionmnist", "mnist"):
        return "cnn"
    return "resnet20"


def build_model(dataset, requested_net, num_outputs=10):
    model_type = resolve_model_type(dataset, requested_net)
    if model_type == "cnn":
        return get_cnn(num_outputs=num_outputs)
    if model_type == "resnet20":
        return get_resnet20(num_outputs=num_outputs, pretrained=False)
    raise NotImplementedError("Unknown model type: %r" % (model_type,))
