# Learned-image-quality-improvement-network
This is a program implemented in PyTorch to implove the image quality.

By executing the program, it is possible to enlarge the image by a factor of 2 to 4.(only 2x at present)

The network architecture is shown below.
![architecture](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/d4c46373-b353-40ea-9fa7-0175ecedfb2e)

This network is trained using LPIPS and MS-SSIM to reconstruct visually superior images.
![スライド2](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/841b48d5-82e4-4327-80e4-f5251236c1cd)
![スライド1](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/2be8ec5c-01a2-47ca-a67d-6f720f5b1493)


## HOW TO USE
Create a folder "Val_Image/test" for input images.
(The use of ImageFolder requires two levels of hierarchy.)
```
CUDA_VISIBLE_DEVICES=0 python train_highres.py -n test -p checkpoints/2x/iter_170000.pth.tar --test
CUDA_VISIBLE_DEVICES=0 python train_highres.py -n test -p checkpoints/4x/iter_155000.pth.tar --test
```
