import logging


def get_logger():
    """ Function to get logger object """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('logs')
