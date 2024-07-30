/******************************************************************************
* Copyright (C) 2023 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file aiert_tests.h
* @{
*
* Header file for declaring Simulator tests (AIEML-Tests)
*
******************************************************************************/

/* List of all tests 
 * Description: Add an external reference to your test-api here
 * */
extern int test_lock(XAie_DevInst *DevInst);

/*
 * Description: Add the function pointer for your test-api
 */
int (*tests_aiert[])(XAie_DevInst *DevInst) =
{
	test_lock
};

/*
 * Description: Add the name of the function here  
 */
const char *test_names_aiert[] =
{
	"test_lock"
};

