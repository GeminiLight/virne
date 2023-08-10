from .tester import Tester
from args import *


class ConfigTester(Tester):

    def __init__(self):
        super(ConfigTester, self).__init__()

    def test_all(self):
        config = get_config()
        show_config(config)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        save_config(config, fname='config_for_test.json')

        
