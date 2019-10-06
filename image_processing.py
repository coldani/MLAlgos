import numpy as np

'''
new_image reduces the definition of an input image to a given size (in terms of number of pixels in the rows and in the columns)
gray_scale converts a RGB image to a Grayscale image
'''

def _downsize_image(img, ratio):
    '''
    this function is called from the "new_image" function

    img: the original image. It must be a numpy ndarray
    ratio: a tuple, or list, with the ratios (input rows)/(output rows) and (input cols)/(output cols)
    '''
    ratio = tuple(ratio)
    row_ratio, col_ratio = ratio

    (input_r, input_c) = img.shape[:2]
    new_r = int(input_r/row_ratio)
    new_c = int(input_c/col_ratio)

    if img.ndim == 3:
        output_img = np.zeros((new_r, new_c, img.shape[2]))
    elif img.ndim == 2:
        output_img = np.zeros((new_r, new_c))
    else:
        print('Check number of dimensions!')


    new_r = int(input_r/row_ratio)
    new_c = int(input_c/col_ratio)
    
    row_indexing = np.reshape(np.linspace(0, new_r*row_ratio, new_r), (new_r, 1)) # mapping from output to input rows
    col_indexing = np.reshape(np.linspace(0, new_c*col_ratio, new_c), (1, new_c)) # mapping from output to input cols
    
    # the variables below are used only for weighting purposes
    rolled_row_indexing = np.roll(row_indexing, -1)
    rolled_row_indexing[-1] = rolled_row_indexing[-2]
    rolled_col_indexing = np.roll(col_indexing, -1)
    rolled_col_indexing[:,-1] = rolled_col_indexing[:,-2]
    ####
    
    # weights
    if img.ndim == 3:
        wr1 = np.reshape((rolled_row_indexing - row_indexing.astype('int') - 1)/2, (new_r, 1, 1))
        wr0 = 1 - wr1
        wc1 = np.reshape((rolled_col_indexing - col_indexing.astype('int') - 1)/2, (1, new_c, 1))
        wc0 = 1 - wc1
    else:
        wr1 = np.reshape((rolled_row_indexing - row_indexing.astype('int') - 1)/2, (new_r, 1))
        wr0 = 1 - wr1
        wc1 = np.reshape((rolled_col_indexing - col_indexing.astype('int') - 1)/2, (1, new_c))
        wc0 = 1 - wc1

    w00 = wr0*wc0
    w10 = wr1*wc0
    w01 = wr0*wc1
    w11 = wr1*wc1
    ####
    
    output_img = output_img + \
                (
                img[np.minimum(row_indexing.astype('int'), input_r-1),np.minimum(col_indexing.astype('int'), input_c-1)]*w00 + \
                img[np.minimum(row_indexing.astype('int')+1, input_r-1),np.minimum(col_indexing.astype('int'), input_c-1)]*w10 + \
                img[np.minimum(row_indexing.astype('int'), input_r-1),np.minimum(col_indexing.astype('int')+1, input_c-1)]*w01 + \
                img[np.minimum(row_indexing.astype('int')+1, input_r-1),np.minimum(col_indexing.astype('int')+1, input_c-1)]*w11 \
                )// \
                (w00 + w10 + w01 + w11)
    
    return output_img

def new_image(img, new_shape):
    '''
    img: the original image. It must be a numpy ndarray
    new_shape: a tuple, or list, with the desired new image shape, in the form of (# rows, # cols)
    
    returns a 'uint8' numpy array with input image reduced to the desired shape.
    '''
    
    if np.max(img) <= 1:
        img = img * 255

    if (np.asarray(new_shape) > np.asarray(img.shape[:2])).any():
        return print("Desired size is bigger than original size")
    
    break_row = False
    break_col = False
    
    output_r = new_shape[0]
    output_c = new_shape[1]

    while True:
        (input_r, input_c) = img.shape[:2]
        
        if output_r < input_r//2:
            row_ratio = 2
        elif input_r == output_r:
            break_row = True
            row_ratio=1
        else:
            row_ratio = input_r/output_r
            
        
        if output_c < input_c//2:
            col_ratio = 2
        elif input_c == output_c:
            break_col = True
            col_ratio=1
        else:
            col_ratio = input_c/output_c
            
        img = _downsize_image(img, (row_ratio, col_ratio))
            
        if (break_row & break_col):
            break
    
    img = img.astype('uint8')
    
    return img

def gray_scale(img):
    '''
    This function converts a rgb image to a grayscale image.
    Returns a 'uint8' array
    '''

    img = img.astype('float')
    weights_array = np.asarray([0.2989, 0.5870, 0.1140]).reshape([1,1,3])
    converted_img = img*weights_array
    converted_img = np.sum(converted_img, axis=2)
    converted_img = converted_img.astype('uint8')

    return converted_img

