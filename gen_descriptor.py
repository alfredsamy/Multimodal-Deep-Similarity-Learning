# from descriptor import *
import pickle
from scipy import misc


# return list(
# 	(pics,
#   id,  
# 	label # denim, pants, ..etc
# 	name # if multiple pics take 1)
# )
def load_pics(path):
	res = []
	with open('file_list.txt') as f:
		for line in f:
			line = line.strip()
			s = line.split('/')
			
			img = misc.imread(path + line)
			
			res += [(img, s[3], s[1] + '/' + s[2], s[4])]
	return res


def gen_features(img, bowDiction_sift, bowDiction_surf):
	f = {}
	f['gist'] = gist_descriptor(img)
	f['sift'] = bow_feature_extract_sift(bowDiction_sift, img)
	f['surf'] = bow_feature_extract_surf(bowDiction_surf, img)
	return f


# features = {
# 	'surf' : []
# 	'sift' : []
# 	'gist' : []
# }

# for img in load_pics():
# 	f = gen_features(img)
# 	features['surf'] += [f['surf']]
# 	features['sift'] += [f['sift']]
# 	features['gist'] += [f['gist']]


# for i in ['surf', 'sift', 'gist']:
# 	save_file = open(i, 'wb')
# 	save = {
# 		i : features[i]
# 	}
# 	pickle.dump(save, save_file, pickle.HIGHEST_PROTOCOL)
# 	save_file.close()


if __name__ == '__main__':
	res = load_pics('./img/')
	for i in range(10):
		print(res[i])