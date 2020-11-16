import json
import cv2
import os
import numpy as np
from skimage.measure import regionprops, label

META = {'image_sz': [], 'patch_sz': [], 'overlap': [], 'patches': []}

class Splitter(object):

    def __init__(self):
        self.patches = None
        self.meta = None

    def patches2D(self, image, patch_sz, overlap):

        img_sz = image.shape
        step = [patch_sz[0]-overlap[0], patch_sz[1]-overlap[1]]

        assert len(patch_sz) == 2
        assert len(overlap) == 2
        assert img_sz[0] >= patch_sz[0] and img_sz[1] >= patch_sz[1]

        patches = {}
        meta = META.copy()
        meta['image_sz'] = img_sz
        meta['patch_sz'] = list(patch_sz)
        meta['overlap'] = list(overlap)
        
        idx0, idx1 = 0, 0 

        while idx0 * step[0] < img_sz[0]:
            if idx0 * step[0] + patch_sz[0] > img_sz[0]:
                Hmin, Hmax = img_sz[0] - patch_sz[0], img_sz[0]
            else:
                Hmin, Hmax = idx0 * step[0], idx0 * step[0] + patch_sz[0]

            idx1 = 0
            while idx1 * step[1] < img_sz[1]:
                if idx1 * step[1] + patch_sz[1] > img_sz[1]:
                    Wmin, Wmax = img_sz[1] - patch_sz[1], img_sz[1]
                else:
                    Wmin, Wmax = idx1 * step[1], idx1 * step[1] + patch_sz[1]
                fname = 'patch-{:03d}-{:03d}'.format(idx0, idx1)
                meta['patches'].append({'name': fname, 'position': [Hmin, Wmin], 'size': [Hmax-Hmin, Wmax-Wmin]})
                patches[fname] = image[Hmin:Hmax, Wmin:Wmax]
                if Wmax == img_sz[1]:
                    break
                else:
                    idx1 += 1

            if Hmax == img_sz[0]:
                break
            else:
                idx0 += 1

        self.patches, self.meta = patches, meta
        return patches, meta

    # def patches3D(self, image, patch_sz, overlap):
    #     pass
    
    def save(self, save_dir):
        if self.meta is not None and self.patches is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'meta.json'), 'w') as outfile:
                json.dump(self.meta, outfile, indent=2)
            for fname, data in self.patches.items():
                cv2.imwrite(os.path.join(save_dir, fname+'.tif'), data)

MATCH_THRES = 0.5
MAX_LABEL_PER_PATCH = 10000
MIN_OVERLAP = 10

class Stitcher(object):

    def __init__(self, save_dir=None):
        self.patches = None
        self.meta = None
        if save_dir is not None:
            self.read_from_folder(save_dir)

    def read_from_folder(self, save_dir):
        if os.path.exists(save_dir) and os.path.isfile(os.path.join(save_dir, 'meta.json')):
            self.patches = {}
            with open(os.path.join(save_dir, 'meta.json')) as f:
                self.meta = json.load(f)
            for f in os.listdir(save_dir):
                if f[-3:] == 'tif':
                    self.patches[f[:-4]] = cv2.imread(os.path.join(save_dir, f), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return self.patches, self.meta

    def stitch2D(self, meta=None, patches=None, mode='average'):
        '''
        Args:
            meta: meta data
            patches: dict of patches
            mode: 'average', 'label'
        '''
        if meta is None or patches is None:
            meta, patches = self.meta, self.patches
        if meta is None or patches is None:
            return None

        if mode == 'raw':
            image = np.zeros(meta['image_sz'])
            for item in self.meta['patches']:
                fname = item['name']
                Hmin, Wmin = item['position'][0], item['position'][1]
                Hmax, Wmax = Hmin + item['size'][0], Wmin + item['size'][1]
                image[Hmin:Hmax, Wmin:Wmax] = image[Hmin:Hmax, Wmin:Wmax] + patches[fname]
            return image


        if mode == 'average':
            image = np.zeros(meta['image_sz'])
            norm = np.zeros(meta['image_sz'][0:2])
            for item in self.meta['patches']:
                fname = item['name']
                Hmin, Wmin = item['position'][0], item['position'][1]
                Hmax, Wmax = Hmin + item['size'][0], Wmin + item['size'][1]
                image[Hmin:Hmax, Wmin:Wmax] = image[Hmin:Hmax, Wmin:Wmax] + patches[fname]
                norm[Hmin:Hmax, Wmin:Wmax] = norm[Hmin:Hmax, Wmin:Wmax] + np.ones(item['size'])
            if norm.ndim != image.ndim:
                norm = np.expand_dims(norm, axis=-1)
            return image/norm
        
        if mode == 'label':
            image = np.zeros(meta['image_sz'][0:2], np.uint16)
            for base, item in enumerate(self.meta['patches']):
                fname = item['name']
                Hmin, Wmin = item['position'][0], item['position'][1]
                Hmax, Wmax = Hmin + item['size'][0], Wmin + item['size'][1]
                patch = patches[fname] + (patches[fname]>0) * MAX_LABEL_PER_PATCH * base
                
                patch_pad = patch.copy()
                HHmin, HHmax = Hmin, Hmax
                WWmin, WWmax = Wmin, Wmax
                if self.meta['overlap'][0] < MIN_OVERLAP:
                    deltaH = MIN_OVERLAP - self.meta['overlap'][0]
                    HHmin, HHmax = max(0, Hmin-deltaH), min(Hmax+deltaH, meta['image_sz'][0])
                    deltaHmin, deltaHmax = Hmin - HHmin, HHmax - Hmax
                    patch_pad = np.pad(patch_pad, ((deltaHmin, deltaHmax),(0,0)), 'edge')
                if self.meta['overlap'][1] < MIN_OVERLAP:
                    deltaW = MIN_OVERLAP - self.meta['overlap'][1]
                    WWmin, WWmax = max(0, Wmin-deltaW), min(Wmax+deltaW, meta['image_sz'][1])
                    deltaWmin, deltaWmax = Wmin - WWmin, WWmax - Wmax
                    patch_pad = np.pad(patch_pad, ((0,0),(deltaWmin, deltaWmax)), 'edge')

                roi= image[HHmin:HHmax, WWmin:WWmax]
                for p in regionprops(roi):
                    labels, counts = np.unique(patch_pad[p.coords[:,0], p.coords[:,1]], return_counts=True)
                    idx = np.argsort(counts)[-1]
                    if labels[idx] != 0 and counts[idx]/p.area > MATCH_THRES:
                        patch[patch==labels[idx]] = p.label
                image[Hmin:Hmax, Wmin:Wmax] = patch
            return label(image)




