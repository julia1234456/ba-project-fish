import cv2

video = cv2.VideoCapture("video_fish.mp4")


total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames = 1800


frame_index = 0
while frame_index < num_frames:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        break
    if frame_index < 1440:
        frame_name = "frame_" + str(frame_index).zfill(6) + ".PNG"
        cv2.imwrite(frame_name, frame)

    frame_index += 1

video.release()