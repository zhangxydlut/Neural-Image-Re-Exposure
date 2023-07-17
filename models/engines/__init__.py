import logging
logger = logging.getLogger('base')


def build_engine(opt):
    model = opt['engine']
    # NIRE
    if model == 'NIRE':
        from models.engines.NIRE_engine import NIRE_Engine
        Engine = NIRE_Engine

    # NAFNet
    elif model == 'NAFNet':
        from models.engines.NAFNet_engine import NAFNetEngine
        Engine = NAFNetEngine

    # NotImplemented
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    engine = Engine(opt)
    logger.info('Engine [{:s}] is successfully built.'.format(engine.__class__.__name__))
    return engine
