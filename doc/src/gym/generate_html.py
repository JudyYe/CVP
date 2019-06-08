example = """    <tr>
  <td align="center" valign="middle"><a href="./src/gym_gif/{:s}.gif"><img src="./src/gym_gif/{:s}.gif" width="500"> </a></td>      
  <td align="center" valign="middle"><a href="./src/gym_gif/{:s}.gif"><img src="./src/gym_gif/{:s}.gif" width="500"> </a></td>
  </tr>
  """

example = """  <tr><td><a href="./src/gym_bs_gif/{:s}.gif"><img src="./src/gym_bs_gif/{:s}.gif" width="1000"> </a></td></tr>
"""
list_file = 'demo_list.txt'

with open(list_file) as fp:
    vid_list = [line.strip() for line in fp]

wr_fp = open('tmp.html', 'w')
for i in range(0, len(vid_list), 2):
    to_str = example.format(vid_list[i], vid_list[i])
    # if i == len(vid_list) - 1:
    #     to_str = example.format(vid_list[i], vid_list[i], vid_list[i], vid_list[i])
    # else:
    #     to_str = example.format(vid_list[i], vid_list[i], vid_list[i + 1], vid_list[i + 1])
    wr_fp.write('%s' % to_str)
wr_fp.close()
