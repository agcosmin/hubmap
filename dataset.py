import torch.utils.data

class HubmapDataset(torch.utils.data.Dataset):
    def __init__(self, path, patch_size):
        super(HubmapDataset).__init__()
        for file in sorted(
            [f for f in os.listdir(path) if f.split('.')[1] == "tiff"]):
            img = util.imread(os.path.join(path, file))

    def __getitem__(self, key):
        pass
