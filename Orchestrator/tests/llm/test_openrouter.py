import os
import unittest
import pytest
from mcpuniverse.llm.openrouter import OpenRouterModel


class TestOpenRouter(unittest.TestCase):

    @pytest.mark.skip
    def test(self):
        model = OpenRouterModel()
        system_message = "As a professional python developer"
        user_message = "please write a program to generate a fibonacci sequence"
        response = model.get_response(system_message, user_message)
        print(response)

    def test_list_undefined_env_vars(self):
        os.environ["OPENROUTER_API_KEY"] = ""
        model = OpenRouterModel()
        r = model.list_undefined_env_vars()
        self.assertListEqual(r, ["OPENROUTER_API_KEY"])

        os.environ["OPENROUTER_API_KEY"] = "xxx"
        model = OpenRouterModel()
        r = model.list_undefined_env_vars()
        self.assertListEqual(r, [])


if __name__ == "__main__":
    unittest.main()
