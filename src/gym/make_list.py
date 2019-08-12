import glob

all_list = glob.glob('*.gif')

fname = 'demo_list.txt'
with open(fname, 'w') as fp:
    for each in all_list:
        index = each.split('_')[0]
        fp.write('%s\n' % index)