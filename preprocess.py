import random

with open('CEDAR/gray_train.txt', 'w') as f:
	for i in range(1, 51):
		for j in range(1, 25):
			for k in range(j+1, 25):
				# f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
				f.write('original_{0}_{1}.png original_{0}_{2}.png 1\n'.format(i, j, k))

		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			# f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))
			f.write('original_{0}_{1}.png forgeries_{0}_{2}.png 0\n'.format(i, j, k))

with open('CEDAR/gray_test.txt', 'w') as f:
	for i in range(51, 56):
		for j in range(1, 25):
			for k in range(j+1, 25):
				# f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
				f.write('original_{0}_{1}.png original_{0}_{2}.png 1\n'.format(i, j, k))
		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			# f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))
			f.write('original_{0}_{1}.png forgeries_{0}_{2}.png 0\n'.format(i, j, k))

