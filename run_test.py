import os

script_path = 'test.py'
how_many = 1000
label_dir = '/data/anpei/facial-data/mask/plain/image/'
image_dir = '/data/anpei/facial-data/images/'
results_dir = './results/Jack12test '
MODE = '1'

for epoch_num in range(2,10,2):
    command = 'python {} ' \
              '--no_instance ' \
              '--use_vae ' \
              '--MODE {} ' \
              '--label_dir {} ' \
              '--image_dir {} ' \
              '--results_dir {} ' \
              '--which_epoch {} ' \
              '--how_many {} '.format(script_path, MODE,label_dir, image_dir, results_dir, epoch_num, how_many)
              
    print("start to run command {}".format(command))
    os.system(command)