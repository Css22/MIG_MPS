QoS_map = {
    "resnet50":45,
    "mobilenet_v2": 40,
    "unet": 60,
    "bert": 200,
    "vgg19": 130,
    "deeplabv3": 150,
}


def get_QoS_map():
    return QoS_map

