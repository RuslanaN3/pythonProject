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


transform_img = transforms.Compose([transforms.ToTensor()])


def get_prediction(img, threshold, smodel):
    img = transform_img(img)  # Apply the transform to the image
    pred = smodel([img])
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding boxes
    pred_labels = list(pred[0]["labels"].detach().numpy())
    pred_scores = list(pred[0]["scores"].detach().numpy())
    pred_th_scores = [round(x, 2) for x in pred_scores if x > threshold]
    pred_t = [pred_scores.index(x) for x in pred_scores if x > threshold][
        -1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_labels = pred_labels[:pred_t + 1]
    return pred_boxes, pred_labels, pred_th_scores
