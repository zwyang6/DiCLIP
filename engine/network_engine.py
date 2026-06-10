from model.model_diclip import DiCLIP_model

def build_network(args):

    model = DiCLIP_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        adapter_size=args.adapter_size, \
                        num_classes=args.num_classes, cache_file=args.cache_file,\
                        img_size=args.crop_size, mode=args.train_set, device='cuda')
    param_groups = model.get_param_groups()

    return model, param_groups