import os
import glob
src_dir = '/tmp/transfer/save_cmp/'
list_file = 'demo_ss.txt'
dst_dir = '../ss_num/'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

with open(list_file) as fp:
    vid_list = [line.strip() for line in fp]

def cp_num():
    for index in vid_list:
        gif_file = os.path.join(src_dir, index + '.gif')
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

example = """    <tr>
  <td align="center" valign="middle"><a href="./src/ss_num/{:s}.gif"><img src="./src/ss_num/{:s}.gif" width="500"> </a></td>      
  <td align="center" valign="middle"><a href="./src/ss_num/{:s}.gif"><img src="./src/ss_num/{:s}.gif" width="500"> </a></td>
  </tr>
  """


def gen_html():
    wr_fp = open('tmp.html', 'w')
    for i in range(0, len(vid_list), 2):
        # to_str = example.format(vid_list[i], vid_list[i])
        if i == len(vid_list) - 1:
            to_str = example.format(vid_list[i], vid_list[i], vid_list[i], vid_list[i])
        else:
            to_str = example.format(vid_list[i], vid_list[i], vid_list[i + 1], vid_list[i + 1])
        wr_fp.write('%s' % to_str)
    wr_fp.close()


cp_num()
# gen_html()
