/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/TestHarness.h"
#include <hw_config.h>

int main(int argc, char**argv)
{
	return CommandLineTestRunner::RunAllTests(argc, argv);
}
