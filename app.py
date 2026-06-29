from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

CHECKPOINT_PATH = Path(__file__).parent / "brain_tumor_vit_model (1).pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["Benign", "Malignant"]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


@st.cache_resource
def load_model() -> nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {CHECKPOINT_PATH}")

    model = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def predict(image: Image.Image, model: nn.Module):
    tensor = preprocess_image(image)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    label_index = int(probabilities.argmax())
    confidence = float(probabilities[label_index])
    return LABELS[label_index], confidence, probabilities.tolist()


def main():
    st.set_page_config(page_title="Brain Tumor ViT Classifier", page_icon="🧠", layout="centered")
    st.title("Brain Tumor Classification")
    st.write("Upload an MRI image and let the ViT model determine whether the tumor is benign or malignant.")

    st.sidebar.title("Instructions")
    st.sidebar.write(
        "1. Upload a brain MRI image file (.jpg, .png, .jpeg, .bmp, .tiff).\n"
        "2. Click the button to analyze the image.\n"
        "3. The app will display benign vs malignant prediction and confidence scores."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Analyze image"):
            try:
                model = load_model()
                label, confidence, probabilities = predict(image, model)
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence * 100:.2f}%")
                st.write(
                    {
                        LABELS[0]: f"{probabilities[0] * 100:.2f}%",
                        LABELS[1]: f"{probabilities[1] * 100:.2f}%",
                    }
                )
            except Exception as exc:
                st.error(f"Unable to run prediction: {exc}")

    st.markdown("---")
    st.markdown("Built with PyTorch and Streamlit.")


if __name__ == "__main__":
    main()
