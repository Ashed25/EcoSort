import timm
import torch

PRETRAINED = {
    # name -> timm id (ImageNet-pretrained CNNs)
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
}


def build_model(arch, num_classes=3, pretrained=True):
    return timm.create_model(PRETRAINED[arch], pretrained=pretrained, num_classes=num_classes, drop_rate=0.3)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), ckpt
