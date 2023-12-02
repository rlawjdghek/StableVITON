import torch.nn as nn
from transformers import CLIPVisionModel
from .xf import LayerNorm, Transformer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                1,
            )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        if isinstance(image, list):
            image = image[0]
        return self(image)

if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPImageEmbedder()
    count_params(model, verbose=True)