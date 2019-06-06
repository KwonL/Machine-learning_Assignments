#!/usr/bin/python3
# Number's size is 24 by 24
# Center is (11, 11) of image(start from 0 to 23)
from data_utils import *
from PIL import Image


def get_data_without_norm(seq) :
    dir='Data/train_sequence/'
    input = np.zeros([len(seq), 20, 64, 64, 3])
    for i, idx in enumerate(seq):
        for t in range(20):
            img_path = os.path.join(dir, 'sequence%04d' % idx, 'frames%02d.png' % t)
            img = np.array(Image.open(img_path))
            input[i, t] = img

    return input


def get_center_of_number(sample_img) :
    r_max = 0
    c_max = 0
    r_min = 63
    c_min = 63

    for r, row in enumerate(sample_img) :
        for c, pixel in enumerate(row) :
            R = pixel[0]

            if R != 0 :
                # print(pixel)
                if c < c_min :
                    c_min = c
                if c > c_max :
                    c_max = c
                if r < r_min :
                    r_min = r 
                if r > r_max :
                    r_max = r

    return (r_max + r_min) // 2, (c_max + c_min) // 2


def main_data2feat() :
    batch_size = 200
    seq = 0
    cordinates_dir = './processed_data/red_num_cordinates/'
    for i in range(int(10000 / batch_size)) :
        print("iter %d" % i)
        batch = get_data_without_norm(list(range(200 * i, 200 * (i + 1))))

        for data_set in batch :
            fd = open(os.path.join(cordinates_dir, '%04d' % seq), 'w')
            # First, extract rednumber
            sample_img = data_set[0]
            x_c, y_c = get_center_of_number(sample_img)
            x_m, y_m = x_c - 11, y_c - 11

            # Extract Image
            num_img = np.zeros([24, 24, 3])
            for i in range(24) :
                for j in range(24) :
                    num_img[i][j][0] = sample_img[x_m + i][y_m + j][0]
            num_img = num_img.astype(np.uint8)

            Image.fromarray(num_img).save('./processed_data/red_num_images/rednum_seq%04d.png' % seq)

            fd.write('%d %d\n' % (x_m, y_m))

            # Then, extract coordinate of numbers
            for idx in range(1, 20) :
                x_tmp, y_tmp = get_center_of_number(data_set[idx])
                fd.write('%d %d\n' % (x_tmp - 11, y_tmp - 11))

            fd.close()
            seq += 1


if __name__ == '__main__' :
    main_data2feat()
