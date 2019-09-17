import glob
path = "./data/"
folders = [f for f in glob.glob(path + "delphes_data?.h5")]
folders += [f for f in glob.glob(path + "delphes_data??.h5")]
folders += [f for f in glob.glob(path + "delphes_data???.h5")]

for f in folders:
    print(f)
print(len(folders))