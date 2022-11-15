# ConchShell: A Generative Adversarial Networks that Turns Pictures into Piano Music  
We present ConchShell, a multi-modal generative adversarial framework that takes pictures as input to the network and generates piano music samples that match the picture context. Inspired by I3D, we introduce a novel image feature representation method: time-convolutional neural network (TCNN), which is used to forge features for images in the temporal dimension. Although our image data consists of only six categories, our proposed framework will be innovative and commercially meaningful. The project will provide technical ideas for work such as 3D game voice overs, short-video soundtracks, and real-time generation of metaverse background music.We have also released a new dataset, the Beach-Ocean-Piano Dataset (BOPD) 1, which contains more than 3,000 images and more than 1,500 piano pieces. This dataset will support multimodal image-to-music research.  
The following are the overall model architecture.

![Model architecture](https://github.com/DIO385/ConchShell/blob/main/modules/ConchShell.jpg)
### Envs
1. Python >= 3.7
2. Clone this repository:
```bash
git https://github.com/yl4579/StarGANv2-VC.git
cd StarGANv2-VC
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio munch parallel_wavegan torch pydub
```
4. Download the [BOPD dataset](https://datashare.ed.ac.uk/handle/10283/3443) 
Then put the dataset into the 'dataset' folder.

## Training
```bash
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=9999 train.py
```
