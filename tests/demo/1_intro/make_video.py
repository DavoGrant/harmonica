import cv2
import os

image_folder = '/Users/davidgrant/Desktop/string_animation/'
video_name = '/Users/davidgrant/Desktop/string_animation/video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = sorted(images, key=lambda x: int(x.split('.')[0]))
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 40, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

