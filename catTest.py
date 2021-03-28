from PIL import Image
import os


count = 13000


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def generate_imgpath(count, imname):
    image_dir = 'test'
    return '%s/%s.png' % (image_dir, imname)

imname_li = ['bg', 'bg_bx', 'fake_img', 'fake_mk', 'fake_bx', 'real', 'real_bx', 'test1', 'test1_bx', 'test2', 'test2_bx', 'test3', 'test3_bx']


im_path = generate_imgpath(count, imname_li[0])
im = Image.open(im_path)
for imname in imname_li[1:]:
    _im_path = generate_imgpath(count, imname)
    _im = Image.open(_im_path)
    im = get_concat_v(im, _im)

res = 128
im.save('test.png')
