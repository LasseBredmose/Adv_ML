import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

from src.models.models import CNN, CNN_nomax


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def cam(image_path, model_path, mp):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mp == 0:
        net = CNN_nomax(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)
    else:
        net = CNN(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)

    last_conv_name = "conv5"

    net.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device),
        )
    )

    net.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(last_conv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load test image
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # load list of classes
    classes = [0, 1, 2, 3, 4, 5, 6]

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    mapper = {
        0: "SHOULDER",
        1: "HUMERUS",
        2: "FINGER",
        3: "ELBOW",
        4: "WRIST",
        5: "FOREARM",
        6: "HAND",
    }

    # output the prediction
    for i in range(0, 5):
        print(
            "{:.3f} -> {} ({})".format(
                probs[i], classes[idx[i]], mapper[classes[idx[i]]]
            )
        )

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print("Output for the top1 prediction: %s" % mapper[classes[idx[0]]])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    hess = model_path.split("_")[1]
    method = model_path.split("_")[2]

    cv2.imwrite(
        f"CAM_{hess}_{method}_mp_{mp}_{'_'.join(image_path.split('/')[3:])}", result
    )

    return CAMs
