# GeodesicDiffusion
Offical implementation for "Probability Density Geodesics in Image Diffusion Latent Space" (CVPR2025)

## Test the code
Create the environment
```bash
conda create -n geodesicdiff python=3.12.4
conda activate geodesicdiff
pip install -r requirements.txt
```

To run the main script, use:
```bash
python test_bvp.py --c configs/config_example.yaml
```

## ToDo 
add the geodesic analysis file
add the ivp test file

## Introduction

This project explores an interesting idea: using an image generation model to create short videos. 

By connecting individual frames, a video can be interpreted as a continuous path through the model's latent space. If this path is a straight line, it may pass through low-probability regions, resulting in poor-quality frames. To address this, we aim for the path to traverse high-probability density regions where the model generates more plausible images.

To achieve this, we optimize the initial straight-line path to better align with these high-density areas. Our approach builds on a simple idea: leveraging the gradient direction provided by score distillation([SDS](https://dreamfusion3d.github.io/),[NFSD](https://orenkatzir.github.io/nfsd/))
, combined with a smoothness constraint derived from a geodesic formulation, to ensure natural and coherent frame transitions.

We implemented this method in a training-free manner using the pre-trained Stable Diffusion 2 model. Although the results do not match the quality of trained video generation models or finetuned continuous image generation models (([DiffMorpher](https://github.com/Kevin-thu/DiffMorpher), [IMPUS](https://github.com/GoL2022/IMPUS), [PAID](https://qy-h00.github.io/attention-interpolation-diffusion/), [SmoothDiffusion](https://github.com/SHI-Labs/Smooth-Diffusion))), our approach provides strong interpretability and analytical tractability.

The theoretical foundation can be traced back to Fermat’s principle([wiki](https://en.wikipedia.org/wiki/Fermat%27s_principle)): light traveling through a medium distribution follows the path of least time. Similarly, we seek generation paths that avoid “resistant” low-probability regions and instead follow “efficient” routes through high-probability areas—resulting in higher-quality frames and smoother temporal transitions.

这个项目提供了一个比较有趣的思路：如何利用图像生成模型来生成短视频。

将每一帧连接起来，视频可以看作是图像生成空间中的一条路径。如果这条路径是直线，可能会经过生成质量较差的区域。因此，我们希望路径能穿过生成模型中的高概率密度区域。

为此，我们对直线路径进行优化，使其更贴近高密度区域。我们试了一个比较基础的想法，通过 score distillation ([SDS](https://dreamfusion3d.github.io/) [NFSD](https://orenkatzir.github.io/nfsd/)) 提供的梯度方向，并加入公式推导出的平滑约束，确保帧间过渡自然、连续。
我们在预训练的 Stable Diffusion 2 上以 training-free 的方式进行了尝试。虽然效果不如训练过的视频生成模型或需要finetune连续图像生成模型([DiffMorpher](https://github.com/Kevin-thu/DiffMorpher), [IMPUS](https://github.com/GoL2022/IMPUS), [PAID](https://qy-h00.github.io/attention-interpolation-diffusion/), [SmoothDiffusion](https://github.com/SHI-Labs/Smooth-Diffusion))，但这个方法具备良好的可解释性和可分析性。

文章的理论的基础可以追溯到费马原理([wiki](https://en.wikipedia.org/wiki/Fermat%27s_principle))，光在一个介质分布中传播时会选择一条使传播时间最短的路径。类似地，我们希望图像生成过程中的路径避开“阻力大”的低概率区域，沿着“通畅”的高概率区域前进，从而生成质量更高、变化更平滑的视频帧。

