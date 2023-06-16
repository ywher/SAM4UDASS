import numpy as np
# import torch
import matplotlib.pyplot as plt
import cv2
from utils.sample_points import uniform_sampling
# import sys
# sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from utils.sample_points import uniform_sampling
from cityscapesscripts.helpers.labels import trainId2label as trainid2label

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
def trainid2color(trainid):
    '''
    function: convert trainID to color in cityscapes
    input: trainid
    output: color
    '''
    #if the input is a number in np.uint8, it means it is a trainid
    if type(trainid) == np.uint8:
        label_object = trainid2label[trainid]
        return label_object.color[::-1]
    else:
        color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
        for i in range(trainid.shape[0]):
            label_object = trainid2label[trainid[i]]
            color_mask[i] = label_object.color[::-1]
    return color_mask
    
def color_segmentation(segmentation):
    #get the color segmentation result, initial the color segmentation result with black (0,0,0)
    #input: segmentation [h, w]
    color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    train_ids = np.unique(segmentation)
    for train_id in train_ids:
        color_segmentation[segmentation == train_id] = trainid2color(train_id)[::-1]
    return color_segmentation


if __name__ == '__main__':
    #set the ids
    thing_ids = range(19)
    # thing_ids = [11, 12, 13, 14, 15, 16, 17, 18]
    
    #load image
    image = cv2.imread('images/aachen_000029_000019_leftImg8bit.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread('images/aachen_000029_000019_gtFine_labelTrainIds.png', cv2.IMREAD_GRAYSCALE)
    color_label = color_segmentation(label)
    
    #show image
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    
    #show label
    plt.figure(figsize=(10,10))
    plt.imshow(color_label)
    plt.show()
    plt.close()

    #load model
    sam_checkpoint = "models/sam_vit_h_4b8939.pth" #sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth
    model_type = "vit_h" #vit_b, vit_h, vit_l

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print('load model successfully')

    #set predictor
    predictor = SamPredictor(sam)
    #Process the image to produce an image embedding by calling `SamPredictor.set_image`. 
    #`SamPredictor` remembers this embedding and will use it for subsequent mask prediction.
    predictor.set_image(image) #get the image embedding, for subsequent mask prediction
    
    #show the points on the image
    # input_point = np.array([[500, 375]]) #input the point prompt in the format of (x, y)
    # input_label = np.array([1]) #input the label prompt in the format of (0: background, 1: foreground)
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()  
    
    # Predict with `SamPredictor.predict`. 
    # The model returns masks, quality predictions for those masks, 
    # and low resolution mask logits that can be passed to the next iteration of prediction.
    
    unique_ids = np.unique(label)
    unique_ids = unique_ids[unique_ids != 255]
    for unique_id in unique_ids:
        if unique_id in thing_ids:
            continue
        id_mask = label == unique_id
        sample_points = uniform_sampling(id_mask, 0.0001, 10)
        input_point = np.array(sample_points)
        input_label = np.array([1] * len(sample_points))
        
    
        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        )
    
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(32,16))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
        plt.close()
        
    # input_point = np.array([[500, 375], [1125, 625]])
    # input_label = np.array([1, 1])

        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        
        masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
        )
    
        plt.figure(figsize=(32,16))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()
    
    # input_point = np.array([[500, 375], [1125, 625]])
    # input_label = np.array([1, 0])

    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
    # masks, _, _ = predictor.predict(
    # point_coords=input_point,
    # point_labels=input_label,
    # mask_input=mask_input[None, :, :],
    # multimask_output=False,
    # )
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show() 