from numpy import zeros,uint,uint8,sqrt,full,append,convolve,insert,linspace,vstack,ones,array
from cv2 import resize,GaussianBlur,line
from scipy.interpolate import splev, splprep

# scale x,y coordinates between 0 and 1 and center them
def scale_and_center(xy):
    xy_copy = xy.copy()

    min_x =  min(xy_copy[:,0])
    max_x =  max(xy_copy[:,0])
    min_y =  min(xy_copy[:,1])
    max_y =  max(xy_copy[:,1])

    max_width = max(max_x-min_x,max_y-min_y)

    xy_copy[:,0] -= min_x # shift coords to that the bottom left is 0
    xy_copy[:,1] -= min_y

    xy_scaled = xy_copy / max_width # scale the image so that the widest final_dimension is 28

    # center the pixels
    xy_scaled[:,0]+= (max_width - (max_x-min_x))/max_width /2
    xy_scaled[:,1]+= (max_width - (max_y-min_y))/max_width /2

    return xy_scaled


def convert_points_to_images(xy_scaled,final_dimension,initial_dimension,line_thickness,border_thickness=1,gaussian_kernel = 21):
    xy = xy_scaled.copy()

    line_thickness = int(line_thickness*initial_dimension/final_dimension)
    border_thickness = int(border_thickness*initial_dimension/final_dimension)
    image = zeros((initial_dimension+2*border_thickness,initial_dimension+2*border_thickness))

    # draw lines between points on the image
    xy = (xy*initial_dimension).astype(uint)
    xy+= border_thickness
    for i in range(len(xy)-1):
        line(image,(xy[i,0],initial_dimension+2*border_thickness-1 - xy[i,1]),(xy[i+1,0],initial_dimension-1+2*border_thickness -xy[i+1,1]),1,line_thickness)
    
    # plt.figure()
    # plt.title(f'Size: {initial_dimension}')
    # plt.imshow(image,"gray")

    if gaussian_kernel!=0:
        image = GaussianBlur(image,(gaussian_kernel,gaussian_kernel),0)

    # plt.figure()
    # plt.title(f'Blurred')
    # plt.imshow(image,"gray")

    image = resize(image,(final_dimension,final_dimension))

    return (image*255).astype(uint8)

def distance(point1, point2):
    return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def remove_close_points(xy_shift):
    theathold_end = 15*280
    theathold_middle = 8*280

    # x,y = xy_shift.T
    # distance_point_next = sqrt((x[:-1]-x[1:])**2+(y[:-1]-y[1:])**2)*100000
    
    # # calculate a severe mark to remove points at the end
    # mask1 = insert(distance_point_next>theathold_end,0,True) # get only distances above 10 + add a value for the 1st point
    # last_consecutive_false_indices = mask1.shape[0]-1
    # while(not mask1[last_consecutive_false_indices]):
    #     last_consecutive_false_indices -= 1
    # last_consecutive_false_indices +=1

    # # calculate another mask to remove duplicated points in the rest 
    # mask = insert(distance_point_next>theathold_middle,0,True) # get only distances above 10 + add a value for the 1st point
    # mask[last_consecutive_false_indices:]=False #force remove the points at the end

    # xy_shift = xy_shift[mask]

    mask = full(xy_shift.shape[0],True) # mask used to reduce the number of points

    tail = True

    filtered_coordinates = xy_shift[-1].reshape(1, -1)  # Start with the first point
    for index in range(1,xy_shift.shape[0]):
        i = xy_shift.shape[0]-1-index

        if tail and distance(xy_shift[i-1], xy_shift[i])*100000 < theathold_end: #compare the the previous for the end
            filtered_coordinates[0]=xy_shift[i] #update the last element
            mask[i]=False
            continue
        else:
            tail=False

        if distance(filtered_coordinates[-1], xy_shift[i])*100000 >= theathold_middle: #compare to the previously save for middle
            filtered_coordinates = append(filtered_coordinates,xy_shift[i].reshape(1, -1),axis=0)
        else:
            mask[i]=False

    filtered_coordinates = filtered_coordinates[::-1]

    return filtered_coordinates,mask

# create a moving average without touching the value that cannot be averaged
def moving_average(a,smooth_value):
    moving_average = convolve(a,ones(smooth_value)/smooth_value,mode='valid')
    moving_average = append(moving_average,a[-(smooth_value//2):]) # add points at the beginning that can't be smoothed
    moving_average = insert(moving_average,0,a[:smooth_value//2]) # add points at the end that can't be smoothed
    return moving_average

def smoothen_points(xy_shift, smooth_value):
    x,y = xy_shift.T
    x,y = moving_average(x,smooth_value),moving_average(y,smooth_value)

    return array([x,y]).T

# from a list of points, returns [X,Y] of the interpolation with the number of points desired
def inter(xy_list,number_points,degree_inter=3):
    tck, _ = splprep(xy_list.T, k=degree_inter, s=0)
    t = linspace(0,1,number_points)
    return vstack(splev(t, tck))
