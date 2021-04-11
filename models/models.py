
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'scagan_pct':
        from .trainer_edge import TransferModel
        model = TransferModel()
    elif opt.model == 'scagan_is':
        from .trainer_pose import TransferModel
        model = TransferModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
