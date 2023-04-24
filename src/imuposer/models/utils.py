from imuposer.models import *

def get_model(config=None, pretrained=None):
    model = config.model
    print(model)

    # load the dataset
    if model == "GlobalModelIMUPoser":
        net = IMUPoserModel(config=config)
    elif model == "GlobalModelIMUPoserFineTuneDIP":
        net = IMUPoserModelFineTune(config=config, pretrained_model=pretrained)
    else:
        print("Enter a valid model")
        return

    return net 
