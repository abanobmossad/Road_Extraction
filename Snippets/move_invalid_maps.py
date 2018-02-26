import os


outputDir = r'E:\Dataset\Test\Test-output-roads'
# changed
output_path = r'E:\mass_roads\Test\Test-output-roads'
input_path = r'E:\mass_roads\Test\Test-output-buildings'


input_images = os.listdir(input_path)

output_images = os.listdir(output_path)

count = 1
for i in output_images:
    name = str(i).split('.')[0]+'.tif'
    if name in input_images:
        print('count',count)
        count +=1
        os.rename(output_path+'/'+i,outputDir+'/'+i)
