from torch.utils.data import Dataset, DataLoader

def make_dataloader(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, drop_last=True)
    return dataloader