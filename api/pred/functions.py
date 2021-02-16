import torchvision.transforms as transforms
from PIL import Image


def transform_image(infile):
    input_transforms = [transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor, model):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction


def render_prediction(prediction_idx):
    return 0
# transform prediction
# stridx = str(prediction_idx)
# class_name = 'Unknown'
# if img_class_map is not None:
#     if stridx in img_class_map is not None:
#         class_name = img_class_map[stridx][1]
#
# return prediction_idx, class_name
