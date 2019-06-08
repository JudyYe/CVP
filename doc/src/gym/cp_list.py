import os
import glob
src_dir = '/tmp/transfer/save_cmp/'
list_file = 'demo_list.txt'
dst_dir = '../gym_bs_gif/'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

with open(list_file) as fp:
    vid_list = [line.strip() for line in fp]

for index in vid_list:
    gif_file = os.path.join(src_dir, index, '*.gif')
    gif_file = glob.glob(gif_file)
    if len(gif_file) != 1:
        print(gif_file)
        continue
    # assert len(gif_file ) == 1, '%d' % len(gif_file)
    gif_file = gif_file[0]

    dst_file = os.path.join(dst_dir, index + '.gif')
    cmd =  "cp '%s' %s" % (gif_file, dst_file)

    print(cmd)
    os.system(cmd)
