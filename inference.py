from model import *

def predict(config, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductRanker(config["l1"], config["l2"]).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']