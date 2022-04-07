import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

from src.models.models import CNN


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        print(weight_softmax[idx].shape)
        cam = weight_softmax.dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def cam(image_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    net.eval()

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

    net._modules.get("conv3").register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load test image
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # load the imagenet category list
    classes = [0, 1, 2, 3, 4, 5, 6]

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print("{:.3f} -> {}".format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print("output CAM.jpg for the top1 prediction: %s" % classes[idx[0]])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite("reports/CAM.jpg", result)
