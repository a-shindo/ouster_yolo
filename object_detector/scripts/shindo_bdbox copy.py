def draw_rect(img,bd_box,dpt_lists):
    for i,(person,dpt_list) in enumerate(zip(bd_box,dpt_lists)):
        cv2.rectangle(
            img,
            pt1=(person[0],person[1]),
            pt2=(person[2],person[3]),
            color=(255,255,255)
            )
        cv2.putText(
          img,
          text=f"P{i} : {str(int(dpt_list[0]))}",
          org=(person[0],person[1]),
          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=1.0,
          color=(125,125,125),
          thickness=2,
          )
    
    return img