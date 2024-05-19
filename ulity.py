import torch.utils.data as Data

def return_batch(train_dataset, batch_size, flag):
    training_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=flag,
        num_workers=0,
    )
    return training_loader

