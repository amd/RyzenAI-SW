/******************************************************************************
* Copyright (C) 2023 AMD, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/***************************** Include Files *********************************/
#include <stdlib.h>
#include <xaiengine.h>

/************************** Constant Definitions *****************************/
/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This function tests AIE Locks
*
* @param	None.
*
* @return	0 on success and error code on failure.
*
* @note		None.
*
*******************************************************************************/
int test_lock(XAie_DevInst *DevInst)
{
	XAie_LocType t = XAie_TileLoc(1,3);
	XAie_LockSetValue(DevInst, t, XAie_LockInit(5,1));
	u32 var;
	AieRC RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,1),&var);
	if(RC != XAIE_OK) {
		printf("[stest/test_lock] XAie_LockGetValue failed.\n");
		return -1;
	}
	
	printf("after set lock value = %x\n",var);
	RC = XAie_LockAcquire(DevInst, t, XAie_LockInit(5, 1),0);
        if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockAcquire failed.\n");
                return -1;
        }

	printf("RC = 0x%x \n",RC);
	RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,1),&var);
        if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockGetValue failed.\n");
                return -1;
        }
	
	printf("after aquire lock value = %x\n",var);
	//RC = XAie_LockRelease(DevInst, t, XAie_LockInit(5,1),0);
	//printf("-Release RC = 0x%x\n",RC);
  	RC = XAie_LockAcquire(DevInst, t, XAie_LockInit(5,1),0);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockAcquire failed.\n");
                return -1;
        }

	printf("after first second aquire RC = 0x%x\n",RC);
	RC = XAie_LockRelease(DevInst, t, XAie_LockInit(5,-1),0);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockRelease failed.\n");
                return -1;
        }

	printf("-Release -1 RC = 0x%x\n",RC);
	RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,1),&var);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockGetValue failed.\n");
                return -1;
        }

	printf("after release lock value = %x\n",var);
  	RC = XAie_LockAcquire(DevInst, t, XAie_LockInit(5,0),0);

	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockAcquire failed.\n");
                return -1;
        }

	printf("after releasea aquire 0x0  RC = 0x%x\n",RC);
	RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,2),&var);
        if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockGetValue failed.\n");
                return -1;
        }

	printf("after second aqc lock value = %x\n",var);


	RC = XAie_LockRelease(DevInst, t, XAie_LockInit(5,1),0);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockRelease failed.\n");
                return -1;
        }

	printf("-Release  5,1 RC = 0x%x\n",RC);
	RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,1),&var);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockGetValue failed.\n");
                return -1;
        }

	printf("after release 5,1 get lock value = %x\n",var);

	RC = XAie_LockRelease(DevInst, t, XAie_LockInit(5,2),0);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockRelease failed.\n");
                return -1;
        }

	printf("-Release 5,2 RC = 0x%x\n",RC);
	RC = XAie_LockGetValue(DevInst, t, XAie_LockInit(5,2),&var);
	if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockGetValue failed.\n");
                return -1;
        }

	printf("after second release get lock value = %x\n",var);

	XAie_LockSetValue(DevInst, t, XAie_LockInit(5,1));
  	RC = XAie_LockAcquire(DevInst, t, XAie_LockInit(5,1),0);
        if(RC != XAIE_OK) {
                printf("[stest/test_lock] XAie_LockAcquire failed.\n");
                return -1;
        }
	
	printf("after set lock RC = 0x%x\n",RC);
	printf("[stest/test_lock] test_lock Passed!\n");
	return 0;
}
