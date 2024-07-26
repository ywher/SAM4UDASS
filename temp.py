# binary mask prompt
mask_to_logits = Segmentix()
# new_mask_input = mask_to_logits.reference_to_sam_mask(bin_mask_input)

bin_masks = {}
center_points = {}

for bin_mask_id in np.unique(label):
    bin_mask_image = np.where(label==bin_mask_id,1,0).astype(np.uint8)
    bin_masks[bin_mask_id] = mask_to_logits.reference_to_sam_mask(np.where(label==bin_mask_id,1,0))

    # 进行连通区域提取
    connectivity = 8  # 连通性，4代表4连通，8代表8连通
    output = cv2.connectedComponentsWithStats(bin_mask_image, connectivity, cv2.CV_32S)

    # 获取连通区域的数量
    num_labels = output[0]

    # 获取连通区域的属性
    labels = output[1]
    stats = output[2]

    cps = []
    
    # 循环遍历每个连通区域
    for i in range(1, num_labels):
        # 获取连通区域的左上角坐标和宽高
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if width * height < 200:
            continue
        
        contours, _ = cv2.findContours(np.uint8(labels == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算区域的质心
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            continue
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # 绘制连通区域的外接矩形
        center_point = (center_x, center_y)

        if bin_mask_image[center_point[1], center_point[0]]:
            cps.append(center_point)
        else:
            points = np.where(labels == i)
            idx = np.random.choice(list(range(len(points[0]))))
            cps.append([points[1][idx], points[0][idx]])
    center_points[bin_mask_id] = cps
    print(center_points)


import os

if not os.path.isdir("./mask_prompt/"):
    os.makedirs("./mask_prompt/")
    print("{} made".format("./mask_prompt/"))
for bin_mask_id in bin_masks:
    print(class_names[bin_mask_id])
    pos = center_points[bin_mask_id]
    neg = []
    for mask_id in center_points:
       if mask_id != bin_mask_id:
           neg += center_points[mask_id]
    
    plt.imshow((np.exp(bin_masks[bin_mask_id])/(np.exp(bin_masks[bin_mask_id])+1))[0], cmap="gray")
    plt.savefig(f"./mask_prompt/type_{class_names[bin_mask_id]}_mask_prompt.png")
    plt.show()
    input_points = np.array(pos + neg)
    input_labels = np.array([1]*len(pos) + [0]*len(neg)),
    mask, _, logit = predictor.predict(
    point_coords= input_points,
    point_labels= input_labes,
    mask_input = bin_masks[bin_mask_id],
    multimask_output=False,
    )
    plt.figure(figsize=(16, 8))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.savefig(f"./mask_prompt/type_{class_names[bin_mask_id]}_image_with_mask.png")
    plt.show()
    plt.close()
    plt.imshow((np.exp(logit)/(np.exp(logit)+1))[0], cmap="gray")
    plt.savefig(f"./mask_prompt/type_{class_names[bin_mask_id]}_logit.png")
    plt.show()
 
 