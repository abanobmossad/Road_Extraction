import os


outputDir = 'E:/mass_roads/train/invalid-map'
output_path = 'E:/mass_roads/train/map'
input_path = 'E:/mass_roads/train/invalid-sat'


input_images = os.listdir(input_path)

output_images = os.listdir(output_path)

count = 1
for i in output_images:
    name = str(i).split('.')[0]+'.tiff'
    if name in input_images:
        print('count',count)
        count +=1
        os.rename(output_path+'/'+i,outputDir+'/'+i)
