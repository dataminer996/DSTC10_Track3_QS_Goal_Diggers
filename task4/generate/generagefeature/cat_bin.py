import pickle
import os
import sys

def get_dict(bin_dir):
    bin_files = os.listdir(bin_dir)
    for bin_file in bin_files:
        with open(os.path.join(bin_dir,bin_file),'rb') as f:
            d = pickle.load(f)
        for i in d:
            yield i

def save_bin(load_dir,save_dir):
    a = list(get_dict(load_dir))
    with open(os.path.join(save_dir,sys.argv[2]), 'wb') as f:
        pickle.dump(a, f)

if __name__ == '__main__':
    bin_dir = sys.argv[1]
    save_bin(bin_dir,'.')
    with open(sys.argv[2], 'rb') as f:
        a = pickle.load(f)
#    print(a)
    print(len(a))
