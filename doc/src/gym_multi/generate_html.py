import os
import glob
import numpy as np

gif_list = glob.glob('*.gif')
gif_list = np.array(sorted(gif_list))
sample_num = 3
gif_list = gif_list.reshape([-1, 3])

vid = len(gif_list)
print(gif_list)

eg = """  <td align="center" valign="middle"><a href="./src/gym_multi/{:s}"><img src="./src/gym_multi/{:s}" width="125"> </a></td>      
"""

wr_fp = open('multi_tmp.html', 'w')
for i in range(sample_num):
    wr_fp.write('<tr>\n')
    for v in range(vid):
        todo = eg.format(gif_list[v, i], gif_list[v, i])
        wr_fp.write('%s' % todo)
    wr_fp.write('</tr>\n')
wr_fp.close()

