from descriptor import bag_of_words_sift, bag_of_words_surf
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


def gen_features(t, bowDiction_sift, bowDiction_surf):
	img = t[0]
	f = {}
	f['gist'] = gist_descriptor(img)
	f['sift'] = bow_feature_extract_sift(bowDiction_sift, img)
	f['surf'] = bow_feature_extract_surf(bowDiction_surf, img)
	f['label'] = t[2]
	return f

print('begin loading pics')
loaded = load_pics('./img/')
print('Loaded Pics')

# fill bowDiction_sift and bowDiction_surf
pics_only = [ loaded[i][0] for i in range(len(loaded)) ]
bag_of_words_sift(pics_only, 1000)
print('Done Sift dict')
bag_of_words_surf(pics_only, 1000)
print('Done Suft dict')


features = {
	'surf' : [],
	'sift' : [],
	'gist' : [],
	'label' : []
}

for t in loaded:
	f = gen_features(t)
	for i in features.keys():
		features[i] += [f[i]]

print('begin saving features to files')

for i in features.keys():
	with open(i + '.pickle', 'wb') as save_file:
		save = {
			i : features[i]
		}
		pickle.dump(save, save_file, pickle.HIGHEST_PROTOCOL)

