# HoloFormer: Contrastive Regularization based Transformer for Holographic Image Reconstruction

Ziqi Bai, [Xianming Liu](https://homepage.hit.edu.cn/xmliu), [Cheng Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=D_jtz9sAAAAJ&view_op=list_works), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Junjun Jiang](https://homepage.hit.edu.cn/jiangjunjun?lang=zh), [Xiangyang Ji](https://www.au.tsinghua.edu.cn/info/1111/1524.htm)

---

Paper link: Waiting for update

---

*In this paper, we introduce HoloFormer, a novel hierarchical framework based on the self-attention mechanism for digital holographic reconstruction. To address the limitation of uniform representation in CNNs, we employ a window-based transformer block as its backbone. Compared to canonical transformer, our structure significantly reduces computational costs, making it particularly suitable for reconstructing high-resolution holograms. To enhance global feature learning capabilities of HoloFormer, a pyramid-like hierarchical structure within the encoder facilitates the learning of feature map representations across various scales. In the decoder, we adopt a dual-branch design to simultaneously reconstruct the real and imaginary parts of the complex amplitude without cross-talk between them. Additionally, supervised deep learning-based hologram reconstruction algorithms typically employ clear ground truth without twin-image for network training, limiting the exploration of contour features in degraded holograms with twin-image artifacts. To address this limitation, we integrate contrastive regularization during the training phase to maximize the utilization of mutual information. Extensive experimental results showcase the outstanding performance of HoloFormer compared to state-of-the-art techniques.*

![Image text](https://github.com/Bzq-Hit/HoloFormer/tree/main/fig/fig.PNG)

---
