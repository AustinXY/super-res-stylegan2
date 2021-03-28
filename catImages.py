from PIL import Image
import os


count = 13000


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def generate_imgpath(count, i):
    image_dir = 'sample'
    return '%s/%06d_%d.png' % (image_dir, count, i)


im_path = generate_imgpath(count, 0)
im = Image.open(im_path)
for i in range(1, 9):
    _im_path = generate_imgpath(count, i)
    _im = Image.open(_im_path)
    im = get_concat_v(im, _im)

res = 128
im.save('%ssample.png' % res)
