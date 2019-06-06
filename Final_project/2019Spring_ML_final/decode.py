import numpy as np
from PIL import Image

def coord_to_image(coord, seq, mode='train') :
    x_red, y_red, x_green, y_green = coord
    red_path = 'processed_data/%s/red_num_images/rednum_seq%04d.png' % (mode, seq)
    green_path = 'processed_data/%s/green_num_images/greennum_seq%04d.png' % (mode, seq)

    img = np.zeros([64, 64, 3])
    red_base_img = np.array(Image.open(red_path))
    green_base_img = np.array(Image.open(green_path))

    print(red_base_img.shape)

    img[x_red:x_red+24, y_red:y_red+24, :] += red_base_img[:, :, :]
    img[x_green:x_green+24, y_green:y_green+24, :] += green_base_img[:, :, :]

    return img


if __name__ == '__main__' :
    print('test mode')
    coord = (3, 4, 40, 31)
    seq = 0000
    img = coord_to_image(coord, seq).astype(np.uint8)
    
    Image.fromarray(img).show()