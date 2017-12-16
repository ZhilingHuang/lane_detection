import moxel

model = moxel.Model('albert/lanedetection:latest', where='localhost')
image = moxel.space.Image.from_file('imgs/image1.png')
result_image = model.predict(img=image)
print(result_image['out'])

