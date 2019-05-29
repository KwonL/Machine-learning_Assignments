#!/usr/bin/python3
# Number's size is 24 by 24
# Center is (11, 11) of image(start from 0 to 23)
from data_utils import *
from PIL import Image
from datetime import datetime
from multiprocessing import Process


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
            G = pixel[1]

            if G != 0 :
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


def get_center_of_number_2(sample_img, base_number_img) :
    max_cnt = 0
    max_offset = 0, 0
    for i in range(64 - 24 + 1) :
        for j in range(64 - 24 + 1) :
            cnt = 0
            for x in range(24) :
                if sample_img[i + x][j + x][1] == base_number_img[x][x][1] :
                    cnt += 1
            if cnt > max_cnt :
                max_cnt = cnt
                max_offset = i + 11, j + 11

    return max_offset


def effective_pixel_num(image) :
    cnt = 0
    for i in range(64) :
        for j in range(64) :
            if image[i][j][1] != 0 :
                cnt += 1

    return cnt


def main_data2feat_2(pid) :
    batch_size = 200
    seq = 1000 * pid
    offset = 1000 * pid
    cordinates_dir = './processed_data/green_num_cordinates/'
    for i in range(int(1000 / batch_size)) :
        print("iter %d" % i)
        time1 = datetime.now()


        batch = get_data_without_norm(list(range(offset + 200 * i, offset + 200 * (i + 1))))

        for data_set in batch :
            fd = open(os.path.join(cordinates_dir, '%04d' % seq), 'w')
            # First, extract rednumber

            # Determine which image is most powerful
            max_idx = 0
            max_num = 0
            for i in range(10) :
                tmp = effective_pixel_num(data_set[i]) 
                if tmp > max_num :
                    max_num = tmp
                    max_idx = i

            sample_img = data_set[max_idx]
            x_c, y_c = get_center_of_number(sample_img)
            x_m, y_m = x_c - 11, y_c - 11

            # Extract Image
            num_img = np.zeros([24, 24, 3])
            for i in range(24) :
                for j in range(24) :
                    num_img[i][j][1] = sample_img[x_m + i][y_m + j][1]
            num_img = num_img.astype(np.uint8)

            Image.fromarray(num_img).save('./processed_data/green_num_images/greennum_seq%04d.png' % seq)

            # Then, extract coordinate of numbers
            for idx in range(20) :
                fd.write('%d %d\n' % get_center_of_number_2(data_set[idx], num_img))

            fd.close()
            print("for seq %d" % seq)
            seq += 1


        time2 = datetime.now()
        print("For one batch, time: " + str(time2 - time1))


if __name__ == '__main__' :
    p_list = list()
    for i in range(10) :
        p = Process(target=main_data2feat_2, args=(i,))
        p_list.append(p)

    for p in p_list :
        p.start()
    
    for p in p_list :
        p.join()