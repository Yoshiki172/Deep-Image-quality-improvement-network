# Learned-image-quality-improvement-network
This is a program implemented in PyTorch to implove the image quality.

By executing the program, it is possible to enlarge the image by a factor of 2 to 4.(only 2x at present)

The network architecture is shown below.
![architecture](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/d4c46373-b353-40ea-9fa7-0175ecedfb2e)

This network is trained using LPIPS and MS-SSIM to reconstruct visually superior images.
![Slide1](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/58c26651-ed88-47ef-a825-d45777336b34)
![Slide2](https://github.com/Yoshiki172/Learned-image-quality-improvement-network/assets/46835185/21de3092-f53e-4c1c-872b-0bd0cafac035)

## HOW TO USE
Create a folder "Val_Image" for input images.
```
CUDA_VISIBLE_DEVICES=0 python train_highres.py -n test -p checkpoints/test/iter_96000.pth.tar
```
