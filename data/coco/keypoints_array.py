import numpy as np

num = 17
KeyPoints = [x for x in range(0, 51)]
matrix = np.array([[1, 2, 3], [4, 5, 6]])


'''skeleton是从json文件导出的骨架点list， matrix是转化矩阵，用源程序中的就可以'''

def crop_hwc(bbox, out_sz = 511, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    # crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          # borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return mapping


def pos_s_2_bbox(pos, s):
	bbox = [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]
    return bbox

def crop_like_SiamFCx(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return target_pos, s_x


def kp_conversion(KeyPoints, matrix, num):

    key_points = []
    kps_conversion = []
    skeleton = [0, 0]
    Skeleton = []

    for i in range(0, num):
        skeleton[0] = KeyPoints[i * 3 + 0]
        skeleton[1] = KeyPoints[i * 3 + 1]
        Skeleton.append(skeleton[:])
        lis = Skeleton[i]
        lis.append(1)
        key_points.append(lis)

    key_points = np.array(key_points)

    for i in range(0, num):
        ky_conversion = np.matmul(matrix, key_points[i, :]).tolist()
        kps_conversion.append(ky_conversion[0])
        kps_conversion.append(ky_conversion[1])
        kps_conversion.append(KeyPoints[i * 3 + 2])

    return kps_conversion


kps_conversion = kp_conversion(KeyPoints, matrix, num)
print(kps_conversion)