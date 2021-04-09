1. sample fineGAN images<br>
``python3 sample_fine_img.py --rand_code c --ds_name LSUNCAR --fine_model ../data/fine_model/lsuncar1600k.pt``<br>
2. project fineGAN images to styleGAN latent space<br>
``python3 projector.py --w_plus  --ckpt ../data/style_model/110000.pt --size 512 fine_sample/fine0.png fine_sample/fine1.png fine_sample/fine2.png fine_sample/fine3.png fine_sample/fine4.png fine_sample/fine5.png fine_sample/fine6.png fine_sample/fine7.png``
