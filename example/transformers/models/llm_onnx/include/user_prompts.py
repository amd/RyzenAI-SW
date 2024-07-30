##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

prompts = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]


prompts_code = [
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# Implement merge sort algorithm.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# Create a function named max_num() that takes a list of numbers named nums as a parameter. The function should return the largest number in nums.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# '写一个函数，找到两个字符序列的最长公共子序列。",
    'from typing import List, Any\n\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    """ Filter given list of any python values only for integers\n    >>> filter_integers(["a", 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, "abc", {}, []])\n    [1, 2, 3]\n    """\n',
    "from typing import List\n\n\ndef concatenate(strings: List[str]) -> str:\n    \"\"\" Concatenate list of strings into a single string\n    >>> concatenate([])\n    ''\n    >>> concatenate(['a', 'b', 'c'])\n    'abc'\n    \"\"\"\n",
    'def max_element(l: list):\n    """Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    """\n',
    'from typing import List, Any\n\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    """ Filter given list of any python values only for integers\n    >>> filter_integers([\'a\', 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, \'abc\', {}, []])\n    [1, 2, 3]\n    """\n',
    "def double_the_difference(lst):\n    '''\n    Given a list of numbers, return the sum of squares of the numbers\n    in the list that are odd. Ignore numbers that are negative or not integers.\n    \n    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10\n    double_the_difference([-1, -2, 0]) == 0\n    double_the_difference([9, -2]) == 81\n    double_the_difference([0]) == 0  \n   \n    If the input list is empty, return 0.\n    '''\n",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Implement quick sort algorithm.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a Python list comprehension to get the squared values of a list [1, 2, 3, 5, 8, 13].",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a Python function that takes in a string and a list of words and returns true if the string contains all the words in the list.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a program to print out the top 3 most frequent words in a given text.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Produce a function that takes two strings, takes the string with the longest length and swaps it with the other.",
]


prompts_chinese = [
    "生命的意义是什么？",
    "告诉我一些你不知道的事情。",
    "Xilinx是做什么的？",
    "地球的质量是多少？",
    "诗歌是什么？",  ##### issue found with this question, qwen1.5 does not reply to this question.
    "递归是什么？",
    "讲一个一句话的笑话.",
    "谁是吉尔伽美什？",
    "告诉我一些关于比特币的事情。",
    "这一切是怎么发生的？",
]
