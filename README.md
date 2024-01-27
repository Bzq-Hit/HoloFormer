# HoloFormer: Contrastive Regularization based Transformer for Holographic Image Reconstruction

Ziqi Bai, [Xianming Liu](https://homepage.hit.edu.cn/xmliu), [Cheng Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=D_jtz9sAAAAJ&view_op=list_works), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Junjun Jiang](https://homepage.hit.edu.cn/jiangjunjun?lang=zh), [Xiangyang Ji](https://www.au.tsinghua.edu.cn/info/1111/1524.htm)

---

Paper link: Waiting for update

---

*In this paper, we introduce HoloFormer, a novel hierarchical framework based on the self-attention mechanism for digital holographic reconstruction. To address the limitation of uniform representation in CNNs, we employ a window-based transformer block as its backbone. Compared to canonical transformer, our structure significantly reduces computational costs, making it particularly suitable for reconstructing high-resolution holograms. To enhance global feature learning capabilities of HoloFormer, a pyramid-like hierarchical structure within the encoder facilitates the learning of feature map representations across various scales. In the decoder, we adopt a dual-branch design to simultaneously reconstruct the real and imaginary parts of the complex amplitude without cross-talk between them. Additionally, supervised deep learning-based hologram reconstruction algorithms typically employ clear ground truth without twin-image for network training, limiting the exploration of contour features in degraded holograms with twin-image artifacts. To address this limitation, we integrate contrastive regularization during the training phase to maximize the utilization of mutual information. Extensive experimental results showcase the outstanding performance of HoloFormer compared to state-of-the-art techniques.*

![Image text](https://github.com/Bzq-Hit/HoloFormer/blob/main/fig/fig.PNG)

---

## Dependencies

For dependencies, you can install them by

```
pip install -r requirements.txt
```

---

## Data

To train HoloFormer, you need to prepare your own hologram dataset, as we do not open-source the dataset used in this research paper. For detailed information about the dataset, please refer to Section Ⅳ.A of the research paper.

Once you have your dataset ready, you can choose to use our provided dataloader by placing the path of your dataset in the appropriate location in `train.py`. This allows you to start training HoloFormer from scratch. Of course, you are also free to develop your own dataloader.

---

## Training

To train HoloFormer, you can begin the training by:

```
python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer
```

To train HoloFormer_S, you can begin the training by:

```
python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer_S
```

To train HoloFormer_T, you can begin the training by:

```
python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer_T
```

---

## Citation

If you find HoloFormer useful in your research, please consider citing:

Waiting for update

---

## Contact

If you have any questions or suggestions regarding this project or the research paper, please feel free to contact the author, Ziqi Bai, at 21B951029@stu.hit.edu.cn.

---


