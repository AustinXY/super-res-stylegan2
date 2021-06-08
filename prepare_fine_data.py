import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os
import pandas as pd

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import functional as trans_fn
import numpy as np

def load_bbox(path):
    # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'
    data_dir = path
    bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    filepath = os.path.join(data_dir, 'images.txt')
    df_filenames = \
        pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        bbox = df_bounding_boxes.iloc[i][1:].tolist()
        key = data_dir + 'images/' + filenames[i]
        filename_bbox[key] = bbox
    return filename_bbox

filename_bbox = None
image_transform = None

def resize_and_convert(img, size, bbox, resample, quality=100):
    if bbox is not None:
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        cimg = img.crop([x1, y1, x2, y2])

        if image_transform is not None:
            img = image_transform(cimg)
    else:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)

    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, bbox=None, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, bbox, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)

    if filename_bbox is not None:
        bbox = filename_bbox[file]

    img = img.convert("RGB")
    out = resize_multiple(img, bbox=bbox, sizes=sizes, resample=resample)

    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]

    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128",
        help="resolutions of images for the dataset",
    )
    parser.add_argument("--resize", action="store_true", help="whether or not resize")
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)
    filename_bbox = load_bbox(args.path)

    if args.resize:
        image_transform = transforms.Compose([
            transforms.Resize(int(sizes[0] * 76 / 64)),
            transforms.RandomCrop(sizes[0])])

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
