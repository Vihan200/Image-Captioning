import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get("step", 0)

def print_examples(model, device, dataset, n=1):
    model.eval()
    for i in range(n):
        img, _ = dataset[i]
        img = img.unsqueeze(0).to(device)
        caption = model.caption_image(img, dataset.vocab)
        print("Predicted Caption:", " ".join(caption))
    model.train()
