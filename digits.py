"""Read actual touchscreen digit images without changing their source files."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONS=('.png','.bmp','.jpg','.jpeg','.pgm')


def preprocess_digit(path):
    """Normalize polarity, crop ink, fit inside 20x20, center in 28x28.

    MNIST has bright digits on a dark background. Border brightness determines
    whether inversion is needed. A preview must still be inspected for real
    hardware data: this is not a general photographic digit segmentation tool.
    """
    with Image.open(path) as source:
        if 'A' in source.getbands():
            rgba=source.convert('RGBA');pixels=np.asarray(rgba)
            opaque=pixels[:,:,:3][pixels[:,:,3]>127]
            background=0 if len(opaque) and opaque.mean()>127 else 255
            canvas=Image.new('RGBA',rgba.size,(background,)*3+(255,))
            image=Image.alpha_composite(canvas,rgba).convert('L')
        else:
            image=source.convert('L')
    array=np.asarray(image,dtype=np.float32)
    border=np.concatenate((array[0],array[-1],array[:,0],array[:,-1]))
    if np.median(border)>127:array=255-array
    array=np.clip(array-float(np.median(np.concatenate((array[0],array[-1],array[:,0],array[:,-1])))),0,255)
    if array.max()<16:raise ValueError('Image has no distinguishable digit: '+str(path))
    points=np.argwhere(array>max(16,float(array.max())*.2))
    if not len(points):raise ValueError('No foreground found: '+str(path))
    y1,x1=points.min(axis=0);y2,x2=points.max(axis=0)+1
    crop=Image.fromarray(array[y1:y2,x1:x2].astype('uint8'))
    scale=min(20/crop.width,20/crop.height)
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(28,28),0)
    canvas.paste(crop,((28-crop.width)//2,(28-crop.height)//2))
    # Match MNIST's center-of-mass convention without wraparound at the edges.
    values=np.asarray(canvas,dtype=np.float32)
    yy,xx=np.indices(values.shape);total=values.sum()
    dx=round(13.5-float((xx*values).sum()/total));dy=round(13.5-float((yy*values).sum()/total))
    canvas=canvas.transform((28,28),Image.Transform.AFFINE,(1,0,-dx,0,1,-dy),resample=Image.Resampling.BILINEAR)
    return np.asarray(canvas,dtype=np.float32)[...,None]/255.0


def load_own_digits(folder):
    """Return None until all ten real digits are present; never invent samples."""
    folder=Path(folder);paths=[];missing=[]
    for digit in range(10):
        found=[folder/('digit_'+str(digit)+ext) for ext in EXTENSIONS if (folder/('digit_'+str(digit)+ext)).is_file()]
        if len(found)>1:raise ValueError('Multiple files for digit '+str(digit))
        if not found:missing.append(digit)
        else:paths.append(found[0])
    if missing:return None,{'status':'pending','missing_digits':missing}
    snapshots={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    images=np.stack([preprocess_digit(p) for p in paths])
    assert snapshots=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return images,{'status':'available','source_sha256':snapshots,'labels':list(range(10))}


def save_preview(images,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    preview=Image.new('L',(28*10,28))
    for digit,image in enumerate(images):
        normalized=Image.fromarray(np.round(image[:,:,0]*255).astype('uint8'))
        normalized.save(folder/('digit_'+str(digit)+'_28x28.png'))
        preview.paste(normalized,(digit*28,0))
    preview.resize((840,84),Image.Resampling.NEAREST).save(folder/'preview_0_to_9.png')
