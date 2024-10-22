import torch
from tqdm.autonotebook import trange, tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


def GAN_invert(G, target_img, z_init=None, lr=1e-2, weight_decay=0e-4, max_iter=5000, print_progress=True, imgsize=256, batch_size=1):
    if z_init is None:
        z_opt = torch.randn(batch_size, 4096, requires_grad=True, device="cuda")
    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        # use MSE loss
        loss = ((img_opt - target_img) ** 2).mean()
        # use L1 loss
        #loss = (img_opt - target_img).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")
        if print_progress:
            print(i, loss.item())
    img_opt = G.visualize(z_opt.detach())
    return z_opt, img_opt

def GAN_invert_with_scheduler(G, target_img, z_init=None, scheduler=None, lr=1e-2, weight_decay=0e-4, max_iter=5000, print_progress=True):
    if z_init is None:
        z_opt = torch.randn(5, 4096, requires_grad=True, device="cuda") 

    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = ExponentialLR(opt, gamma=0.999)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        loss = ((img_opt - target_img) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
        pbar.set_description(f"loss: {loss.item():.3f}, lr: {scheduler.get_last_lr()[0]:.3e}")
        if print_progress:
            print(i, loss.item(), "lr", scheduler.get_last_lr()[0])
    img_opt = G.visualize(z_opt.detach())
    return z_opt, img_opt

def GAN_invert_with_feature_loss(G, target_img, z_init=None, lr=1e-2, weight_decay=0e-4, max_iter=5000, print_progress=True, imgsize=256, batch_size=1):
    from core.utils.image_similarity import TorchImageDistance
    img_dist_obj = TorchImageDistance()
    img_dist_obj.set_first_image_batch(target_img.cpu())
    if z_init is None:
        z_opt = torch.randn(batch_size, 4096, requires_grad=True, device="cuda")
    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        img_dist_obj.set_second_image_batch(img_opt.cpu())
        feature_distance,_ = img_dist_obj.get_CCN_distance(units_slice='all')
        feature_distance = torch.tensor(feature_distance, requires_grad=True, device="cuda")
    
        # use only feature loss 
        loss = feature_distance
        # use L1 loss
        #loss = (img_opt - target_img).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")
        if print_progress:
            print(i, loss.item())

    img_dist_obj.memory_cleanup() 
    img_opt = G.visualize(z_opt.detach())
    
    return z_opt, img_opt

def GAN_invert_with_feature_and_image_loss(G, target_img, z_init=None, lr=1e-2, weight_decay=0e-4, max_iter=5000, print_progress=True, imgsize=256, batch_size=1, feat_landa=.5, img_landa=.5):
    from core.utils.image_similarity import TorchImageDistance
    img_dist_obj = TorchImageDistance()
    img_dist_obj.set_first_image_batch(target_img.cpu())
    if z_init is None:
        z_opt = torch.randn(batch_size, 4096, requires_grad=True, device="cuda")
    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        img_dist_obj.set_second_image_batch(img_opt.cpu())
        # let get the feature distance
        feature_distance,_ = img_dist_obj.get_CCN_distance(units_slice='all')
        feature_distance = torch.tensor(feature_distance, requires_grad=True, device="cuda")

        #let get the MSE
        MSE_dist = ((img_opt - target_img) ** 2).mean()

        # combine the losses        
        loss = feat_landa * feature_distance + img_landa * MSE_dist

        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")
        if print_progress:
            print(i, loss.item())

    img_dist_obj.memory_cleanup() 
    img_opt = G.visualize(z_opt.detach())
    
    return z_opt, img_opt