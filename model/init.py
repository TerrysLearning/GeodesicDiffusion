from scipy.interpolate import CubicSpline
import argparse
import yaml
import os
import shutil
import time
import copy

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)


# for lora fine-tuning
# from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
# from accelerate import Accelerator
# import peft
# import itertools
# from diffusers.optimization import get_scheduler
# from pathlib import Path
# from accelerate.utils import set_seed
# from diffusers.utils.import_utils import is_xformers_available
# from contextlib import nullcontext
# import torch.nn.functional as F
# from peft import LoraConfig, get_peft_model, PeftModel
# from torch.utils.data import Dataset
# import transformers
# from torchvision import transforms

'''
pip install numpy
pip install pillow
pip install diffusers
pip install transformers
pip install scipy
pip install seaborn
pip install scikit-learn
pip install git+https://github.com/patrick-kidger/torchcubicspline.git
pip install accelerate
pip install lpips

** A future warning
/home/u6839880/miniconda3/envs/diffusion0/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
'''