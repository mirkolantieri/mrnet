import torch
from torchvision import models, transforms


class Resnet18(torch.nn.Module):
    """
    class `SimilarRes18`: the implementation of the ResNet18 modern convolutional neural network \\
    to be used for the detection of similarity between the complex cases and other
    cases from the dataset. \\
    Weights initialization with "Imagenet"
    """

    def __init__(self):
        super().__init__()

        # for this prototype we use no gpu, cuda= False to obtain feature vectors
        self.device = torch.device("cpu")

        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.get_feature_layer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_feature_array(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copy_data(m, i, o):
            embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copy_data)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def get_feature_layer(self):
        model = models.resnet18(pretrained=True)
        layer = model._modules.get("avgpool")
        self.layer_output_size = 512

        return model, layer
