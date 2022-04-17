import unittest

from config import get_config


class Tester(unittest.TestCase):
    
    def __init__(self):
        self.config = get_config()

    def test_all(self):
        pass