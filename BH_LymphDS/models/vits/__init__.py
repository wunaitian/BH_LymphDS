import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()
from BH_LymphDS.models.vits.CvT_model import ConvolutionalVisionTransformer # cvt-13_384-IN-22k.
from BH_LymphDS.models.vits.swin_model import SwinTransformer # swin_base_patch32_window7_224_in22k
