1. sample fineGAN images<br>
``python3 sample_fine_img.py --rand_code c --ds_name LSUNCAR --fine_model ../data/fine_model/lsuncar1600k.pt``<br><br>
2. project fineGAN images to styleGAN latent space<br>
``python3 projector.py --w_plus --ckpt ../data/style_model/110000.pt --size 512 fine_sample/fine0.png fine_sample/fine1.png fine_sample/fine2.png fine_sample/fine3.png fine_sample/fine4.png fine_sample/fine5.png fine_sample/fine6.png fine_sample/fine7.png``<br><br>
3. project fineGAN images using PULSE projector<br>
``python3 pulse_projector.py --ckpt ../data/style_model/110000.pt --size 512 --steps 1000 fine_sample/fine0.png fine_sample/fine1.png fine_sample/fine2.png fine_sample/fine3.png fine_sample/fine4.png fine_sample/fine5.png fine_sample/fine6.png fine_sample/fine7.png``<br><br>
4. train mapping net<br>
``python3 train.py --batch 8 --size 512 --style_model ../data/style_model/lsuncar_512_120k.pt --fine_model ../data/fine_model/lsuncar1600k.pt ../data/LMDB/lsun100k_lmdb/``<br><br>
5. train in distribute setting<br>
``python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=8005 train.py --batch 4 --size 512 --style_model ../data/style_model/lsuncar_512_120k.pt --fine_model ../data/fine_model/lsuncar1600k.pt ../data/LMDB/lsun100k_lmdb/``<br><br>
7. test latent 1<br>
test change real sampled image color using latent code -+ fine-mapped latents of same context different color<br>
``python3 test_latent1.py --ckpt ../data/mp_model/lsuncar_120k.pt``<br><br>
___
cdn<br>
decomposer/composer: decompose w code into invariant and variant parts<br><br>
cdn 1<br>
Implicit mixer: take 2 w's as input and output mixed w<br><br>
cdn 2<br>
distiller/mixer: distill variance encoding and mix with w<br><br>
cdn 4<br>
Image mixer: take 2 images as input and output mixed w<br><br>
cdn 4_1<br>
Image mixer: with background preservation<br><br>
cdn 4_2<br>
Image mixer: stitch shape bg to mixed fg<br><br>
cdn 4_3<br>
Image mixer: No mpnet; just require mix image to have same background as shape img and same mean color as color img?<br><br>
___
train<br>
ragular mapping net<br><br>
train1<br>
respect stylegan syntax: only train on stylegan downsample to minimize wp mse<br><br>
train2<br>
create syntax: only train on stylgan downsample but only inflict image mse. A discriminator will also need to be trained.
