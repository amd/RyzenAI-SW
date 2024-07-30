import re
import os
import sys
import argparse
import shutil


home_dir=os.getcwd()
X86_CLIEMB=home_dir+"/CLIEMB/bin/x86/cliemb"
X86_CLIEMB_LIBS=home_dir+"/CLIEMB/src/aie2_driver/lib_x86/"
OUTPUT_DIR=home_dir+"/hw_package/workspace/"
RUN_MODE=""
FORMAT=""
TARGET="STX"
print("HOME DIRECTORY PATH   :- "+home_dir)
print("X86 CLIEMB BIN PATH   :- "+X86_CLIEMB)
print("X86 CLIEMB LIBS PATH  :- "+X86_CLIEMB_LIBS)
print("OUTPUT DIRECTORY PATH :- "+OUTPUT_DIR)
axi_mm_dump_temp_fldr=home_dir+"/aiesimulator_output/"
axi_mm_dump_fldr=home_dir+"/aiesimulator_output/"
xsdb_format_fldr=OUTPUT_DIR+"/XSDB_format/"
array_io_format_fldr=OUTPUT_DIR+"/ARRAY_IO_format/"
dpu_sequence_fldr=OUTPUT_DIR
amd_phoenix_fldr=OUTPUT_DIR+"/PSS_SEQUENCE/"
IS_POWER=False
tcl_file_path=""

tcl_file_name=""

SHIM_BD_LENGTH=8
MEM_BD_LENGTH=8
CORE_BD_LENGTH=6

SHIM_ROW = 0
MEM_ROW_START = 1
MEM_ROW_END = 1
CORE_ROW_START = 2
COL_START=0
NUM_COLS=4
NUM_CORE_ROWS=4

NOC_ADDRESS_OFFSET=0x10000000
PSS_NOC_ADDRESS_OFFSET=0x10000000
DDR_ADDRESS=0x80000000
SHIM_NOC_REG_RANGE0=range(0x14000,0x1F004,0x4)
SHIM_NOC_REG_RANGE1=range(0x14000,0x43FFC,0x4)
SHIM_PL_REG_RANGE0=range(0x31000,0x3FF38,0x4)
SHIM_PL_REG_RANGE1=range(0xFFF00,0xFFF40,0x4)
MEM_DMA_REG_RANGE=range(0xA0600,0xA0660,0x4)
MEM_DMA_STATUS_REG_RANGE=range(0xA0660,0xA0698,0x4)
MEM_LOCK_REG_RANGE=range(0xC0000,0xC03F4,0x4)
SHIM_DMA_REG_RANGE=range(0x1D200,0x1D220,0x4)
SHIM_BD_BURST_REG=range(0x1D028,0x1D200,0x20)
SHIM_DMA_STATUS_REG_RANGE=range(0x1D220,0x0001D228,0x4)
dpu_opcodes = {
        "OP_NOOP" : 0,
        "OP_WRITE32" : 2,
        "OP_SYNC" : 3,
        "OP_READ32_CMP" : 14,
        "OP_READ32_POLL" : 15,
        "OP_WRITESHIMBD" : 11,
        "OP_WRITEMEMBD" : 12
        #define OP_SYNC 3
        #define OP_WRITEBD_EXTEND_AIETILE 4
        #define OP_WRITE32_EXTEND_GENERAL 5
        #define OP_WRITEBD_EXTEND_SHIMTILE 6
        #define OP_WRITEBD_EXTEND_MEMTILE 7
        #define OP_WRITE32_EXTEND_DIFFBD 8
        #define OP_WRITEBD_EXTEND_SAMEBD_MEMTILE 9
        #define OP_DUMPDDR 10
        #define OP_WRITESHIMBD 11
        #define OP_WRITEMEMBD 12
        }

def createDir(folderName):
    if os.path.exists(OUTPUT_DIR):
        print("Folder : "+OUTPUT_DIR+" Already Exists, so using that")
    else:
        os.mkdir(OUTPUT_DIR)

    if os.path.exists(folderName):
        print("Folder : "+folderName+" Already Exists, so using that")
    else:
        os.mkdir(folderName)

def amd_phoenix_pss_dump_new():
    createDir(dpu_sequence_fldr)
    file_write=open(dpu_sequence_fldr+"pss_sequence.txt", "w")
    addressing_mode = 0
    subtract_val = hex ( 0x20000000000 + ((0) << 25))
    write_reg = []
    write_val = []
    os.system("rm -rf temp.txt")
    os.system("grep W "+ axi_mm_dump_temp_fldr+"aiesim_debug_axi_mm_dump.txt" + " > temp.txt")
    mask_writes = {}
    #with open(axi_mm_dump_temp_fldr+"dump_"+tcl_file_name+".txt") as file:
    if IS_POWER == True:
        file_write.write("int col; \n")
        file_write.write("int row; \n")
        file_write.write("for( col = "+str(COL_START)+"; col < "+str(NUM_COLS)+"; col++) {\n")
        file_write.write("    for( row = "+str(CORE_ROW_START)+"; row < "+str(CORE_ROW_START+NUM_CORE_ROWS)+"; row++) {\n")
    with open("temp.txt","r") as file:        
        for line in file:
            print(line)
            if "check_perf" in line:
                file_write.write(line)
                sys.exit(1)
            if "W " in line:
                #if "MW:" in line:
                #    address=line.split(":")[1].split(",")[0].strip("\n")
                #    value=line.split(":")[1].split(",")[2].strip("\n")
                #    mask_value=line.split(":")[1].split(",")[1].strip("\n")
                #    row=(int(address,0) & 0x1F00000) >> 20
                #    col=(int(address,0) & 0xFE000000) >> 25
                #    addr_offset=(int(address,0) & 0xFFFFFFFF)
                #    if len(mask_writes) == 0 or (str(hex(addr_offset)) not in mask_writes or ((int(address,0)&0x00FFFFF) == 0x32000)):  
                #        mask_writes[str(hex(addr_offset))] = value
                #        if IS_POWER == True:
                #            file_write.write("      *(volatile unsigned int*)((col<<25)+(row<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) ="+str(value)+","+str(mask_value)+" ;\n")
                #        else:
                #            file_write.write("*(volatile unsigned int*)(("+str(col)+"<<25)+("+str(row)+"<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) ="+str(value)+","+str(mask_value)+" ;\n")
                #    else:
                #        old_value = int(mask_writes[str(hex(addr_offset))],0)
                #        new_value = hex(old_value | int(value ,0))
                #        mask_writes[str(hex(addr_offset))] = new_value
                #        value = new_value
                #        if IS_POWER == True:
                #            file_write.write("     *(volatile unsigned int*)((col<<25)+(row<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) = "+str(value)+" ;\n")
                #        else:
                #            file_write.write("*(volatile unsigned int*)(("+str(col)+"<<25)+("+str(row)+"<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) = "+str(value)+" ;\n")
                #else:
                address=line.split(":")[1].split(",")[0].strip("\n")
                value=line.split(":")[1].split(",")[1].strip("\n")
                addr_offset=(int(address,0) & 0xFFFFFFFF)                    
                ddr_address_check = addr_offset & 0xFF000000
                row=(int(address,0) & 0x1F00000) >> 20
                col=(int(address,0) & 0xFE000000) >> 25
                if ddr_address_check == DDR_ADDRESS:
                    addr_offset=0x54000000 + (addr_offset & 0x00FFFFFF)
                    file_write.write("*(volatile unsigned int*)(0x"+'{:08x}'.format(addr_offset)+") ="+str(value)+" ;\n")
                else:
                    if IS_POWER == True:
                        file_write.write("     *(volatile unsigned int*)((col<<25)+(row<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) ="+str(value)+" ;\n")
                    else:
                        file_write.write("*(volatile unsigned int*)(("+str(col)+"<<25)+("+str(row)+"<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+")) ="+str(value)+" ;\n")
            #if "R:" in line:
            #    address=line.split(":")[1].split(",")[0].strip("\n")
            #    value=line.split(":")[1].split(",")[1].strip("\n")
            #    row=(int(address,0) & 0x1F00000) >> 20
            #    col=(int(address,0) & 0xFE000000) >> 25
            #    addr_offset=(int(address,0) & 0xFFFFFFFF)
            #    ddr_address_check = addr_offset & 0xFF000000
            #    if ddr_address_check == DDR_ADDRESS:
            #        addr_offset=0x54000000 + (addr_offset & 0x00FFFFFF)
            #        file_write.write("read_compare(0x"+'{:08x}'.format(addr_offset)+","+str(value)+") ;\n")
            #    else:
            #        if IS_POWER == True:
            #            file_write.write("      read_compare((col<<25)+(row<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+"),"+str(value)+") ;\n")
            #        else:
            #            file_write.write("read_compare(("+str(col)+"<<25)+("+str(row)+"<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+"),"+str(value)+") ;\n")
            #        #file_write.write("read_compare(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+addr_offset)+","+str(value)+") ;\n")
            #if "MP:" in line:
            #    address=line.split(":")[1].split(",")[0].strip("\n")
            #    value=line.split(":")[1].split(",")[2].strip("\n")
            #    mask_value=line.split(":")[1].split(",")[1].strip("\n")
            #    addr_offset=(int(address,0) & 0xFFFFFFFF)
            #    row=(int(address,0) & 0x1F00000) >> 20
            #    col=(int(address,0) & 0xFE000000) >> 25
            #    timeout="20"
            #    #file_write.write("read_mask_poll(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+addr_offset)+","+str(mask_value)+","+str(value)+",0) ;\n")
            #    if IS_POWER==True:
            #        file_write.write("    read_mask_poll((col<<25)+(row<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+"),"+str(mask_value)+","+str(value)+",0) ;\n")
            #    else:
            #        file_write.write("read_mask_poll(("+str(col)+"<<25)+("+str(row)+"<<20)+(0x"+'{:08x}'.format(PSS_NOC_ADDRESS_OFFSET+(addr_offset&0xFFFFF))+"),"+str(mask_value)+","+str(value)+",0) ;\n")
    if IS_POWER == True:
        file_write.write("    }\n")
        file_write.write("}\n")

    file_write.close()   

def dpu_sequence_dump():
    ddr_type = 0
    aiesim_ifm_ddr_base_addr=0x0
    aiesim_param_ddr_base_addr=0x0
    aiesim_ofm_ddr_base_addr=0x0
    createDir(dpu_sequence_fldr)
    dump_frmt=open(dpu_sequence_fldr+"mc_code.txt", "w")
    mask_writes={}
    s2mm_bds={}
    mm2s_bds={}
    mem_mm2s_poll_dict={}
    shim_tile_dma_poll=0
    shim_tile_dma_poll_col=[]
    ####################Parse the shim BD offsets######################
    shim_bd_addr_offsets={}
    with open("HW_BO_offsets.txt", "r") as BO_offsets:
        for line in BO_offsets:
            key, value = line.rstrip("\n").split("=")
            shim_bd_addr_offsets[key] = value
    print(shim_bd_addr_offsets)
    #Reset all the DMA channels of mem tile
    for reset_col in range(4):
        for reset_reg_addr in range(0xA0600,0xA0660,0x8):
            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(reset_col)+" ROW : "+str(1)+"\n")
            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(reset_reg_addr))+"\n")
            dump_frmt.write("# VALUE TO WRITE : "+str(2)+"\n")
            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(reset_col))+'{:02x}'.format(int(1))+"00\n")
            dump_frmt.write('{:08x}'.format(int(reset_reg_addr))+"\n")
            dump_frmt.write('{:08x}'.format(2)+"\n")
            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(reset_col)+" ROW : "+str(1)+"\n")
            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(reset_reg_addr))+"\n")
            dump_frmt.write("# VALUE TO WRITE : "+str(0)+"\n")
            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(reset_col))+'{:02x}'.format(int(1))+"00\n")
            dump_frmt.write('{:08x}'.format(int(reset_reg_addr))+"\n")
            dump_frmt.write('{:08x}'.format(0)+"\n")
    with open(axi_mm_dump_fldr+"aiesim_debug_axi_mm_dump.txt") as file:
        for line in file.readlines():
            if "tile_addr=0x1d204" in line or "tile_addr=0x1d20C" in line:
                bdnum = int(line.split(" data=")[1].split(" ")[0].strip("\n"),0) & 0xF
                s2mm_bds[str(bdnum)] = str(bdnum)
            if "tile_addr=0x1d214" in line or "tile_addr=0x1d21C" in line:
                bdnum = int(line.split(" data=")[1].split(" ")[0].strip("\n"),0) & 0xF
                mm2s_bds[str(bdnum)] = str(bdnum)
        file.seek(0)
        for line in file:
            IS_MEM=False
            IS_SHIM=False
            IS_CORE=False
            if "W " in line:
                #if "MW:" in line:
                #    address=line.split(":")[1].split(",")[0].strip("\n")
                #    value=line.split(":")[1].split(",")[2].strip("\n")
                #    row=(int(address,0) & 0x1F00000) >> 20
                #    col=(int(address,0) & 0xFE000000) >> 25
                #    addr_offset=(int(address,0) & 0x000FFFFF)
                #    if len(mask_writes) == 0 or (str(hex(addr_offset)) not in mask_writes):  
                #        mask_writes[str(hex(addr_offset))] = value
                #    else:
                #        old_value = int(mask_writes[str(hex(addr_offset))],0)
                #        new_value = hex(old_value | int(value ,0))
                #        mask_writes[str(hex(addr_offset))] = new_value
                #        value = new_value
                #    dump_frmt.write("# MASK WRITE OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                #    dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                #    dump_frmt.write("# CALUE TO WRITE : "+str(value)+"\n")
                #    dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                #    dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                #    dump_frmt.write('{:08x}'.format(int(value,0))+"\n")                    
                #else:
                address=line.split(" addr=")[1].split(" ")[0].strip("\n")
                value=line.split(" data=")[1].split(" ")[0].strip("\n")
                row=(int(address,0) & 0x1F00000) >> 20
                col=(int(address,0) & 0xFE000000) >> 25
                addr_offset=(int(address,0) & 0x000FFFFF)                    
                if row == SHIM_ROW:
                    BD_Range = range(0x1D000, 0x1D1FC,0x4)
                    BD_LENGTH = SHIM_BD_LENGTH
                    INIT_BD_ADDR=range(0x1D000,0x1D1FC,4*BD_LENGTH)
                    IS_SHIM=True
                if row >= MEM_ROW_START and row <= MEM_ROW_END:
                    BD_Range = range(0xA0000, 0xA05FC,0x4)
                    BD_LENGTH = MEM_BD_LENGTH
                    INIT_BD_ADDR=range(0xA0000,0xA05FC,4*BD_LENGTH)
                    IS_MEM=True
                if row >= CORE_ROW_START :
                    BD_Range = range(0x1D000, 0x1D1F4,0x4)
                    BD_LENGTH = CORE_BD_LENGTH
                    INIT_BD_ADDR=range(0x1D000,0x1D1F4,(4*BD_LENGTH))
                    IS_CORE=True                    
                if addr_offset in BD_Range and addr_offset in INIT_BD_ADDR and IS_CORE == False:
                    BD_NUM=INIT_BD_ADDR.index(int(addr_offset))     
                    bd_len=0
                    if IS_SHIM == True:
                        if str(BD_NUM) in s2mm_bds:
                            ddr_type=2
                        if str(BD_NUM) in mm2s_bds:
                            ddr_type=0
                        print("DDR_TYPE : "+ str(ddr_type))
                        print(str(BD_NUM))
                        print(s2mm_bds)
                        print(mm2s_bds)
                        if (BD_NUM==0):
                            dump_frmt.write("# OPCODE : OP_WRITESHIMBD COL : "+str(col)+" ROW : "+str(row)+" DDR_TYPE :"+str(0)+" BD : "+str(BD_NUM)+"\n")                                 
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITESHIMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:01x}'.format(int(0))+'{:01x}'.format(int(BD_NUM))+"\n")
                        elif (BD_NUM==1):
                            dump_frmt.write("# OPCODE : OP_WRITESHIMBD COL : "+str(col)+" ROW : "+str(row)+" DDR_TYPE :"+str(1)+" BD : "+str(BD_NUM)+"\n")                                 
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITESHIMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:01x}'.format(int(1))+'{:01x}'.format(int(BD_NUM))+"\n")
                        elif (BD_NUM==2):
                            dump_frmt.write("# OPCODE : OP_WRITESHIMBD COL : "+str(col)+" ROW : "+str(row)+" DDR_TYPE :"+str(2)+" BD : "+str(BD_NUM)+"\n")                                 
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITESHIMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:01x}'.format(int(2))+'{:01x}'.format(int(BD_NUM))+"\n")
                        elif (BD_NUM==3):
                            dump_frmt.write("# OPCODE : OP_WRITESHIMBD COL : "+str(col)+" ROW : "+str(row)+" DDR_TYPE :"+str(0)+" BD : "+str(BD_NUM)+"\n")                                 
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITESHIMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:01x}'.format(int(0))+'{:01x}'.format(int(BD_NUM))+"\n")
                        elif (BD_NUM==4):
                            dump_frmt.write("# OPCODE : OP_WRITESHIMBD COL : "+str(col)+" ROW : "+str(row)+" DDR_TYPE :"+str(1)+" BD : "+str(BD_NUM)+"\n")                                 
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITESHIMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:01x}'.format(int(1))+'{:01x}'.format(int(BD_NUM))+"\n")
                        dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                    if IS_MEM == True:
                        dump_frmt.write("# OPCODE : OP_WRITEMEMBD COL : "+str(col)+" ROW : "+str(row)+" BD : "+str(BD_NUM)+"\n")
                        dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                        dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITEMEMBD']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:02x}'.format(int(BD_NUM))+"\n")
                        dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                    bd_len = bd_len+1
                elif addr_offset in BD_Range and IS_CORE == False and bd_len < BD_LENGTH:
                    print("value = "+ str(value))
                    if IS_SHIM == True and bd_len == 1:#Subtract the base address of aiesim buffer to retain the offset in the address field of the BD
                        if (BD_NUM==0):
                            dump_frmt.write('{:08x}'.format(int(shim_bd_addr_offsets['matA_'+str(col)],0))+"\n")
                        elif (BD_NUM==1):
                            dump_frmt.write('{:08x}'.format(int(shim_bd_addr_offsets['matB_'+str(col)],0))+"\n")
                        elif (BD_NUM==2):
                            dump_frmt.write('{:08x}'.format(int(shim_bd_addr_offsets['matC_'+str(col)],0))+"\n")
                        elif (BD_NUM==3):
                            dump_frmt.write('{:08x}'.format(int(shim_bd_addr_offsets['instr_'+str(col)],0))+"\n")
                        elif (BD_NUM==4):
                            dump_frmt.write('{:08x}'.format(int(shim_bd_addr_offsets['matB2_'+str(col)],0))+"\n")
                    elif IS_SHIM == True and bd_len == 4:#Modify the default Burst length of the shim BD field to 11
                        dump_frmt.write('{:08x}'.format(int(value,0) | 0xC0000000)+"\n")
                    else:
                        dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                    bd_len = bd_len+1
                else:
                    if IS_SHIM == True:
                        if addr_offset in SHIM_DMA_REG_RANGE:
                            print("Shim tile DMA config")
                            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                            if col not in shim_tile_dma_poll_col:
                                shim_tile_dma_poll_col.append(col)
                        elif addr_offset in SHIM_NOC_REG_RANGE1 or addr_offset in SHIM_NOC_REG_RANGE0 or addr_offset in SHIM_PL_REG_RANGE0 or addr_offset  in SHIM_PL_REG_RANGE1:
                            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                            print("Shim tile noc reg range")
                    elif IS_MEM == True:
                        if addr_offset in MEM_DMA_REG_RANGE:
                            mem_mm2s_poll_dict={}
                            shim_tile_dma_poll=1
                            print("Mem tile DMA config")
                            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                        elif addr_offset in MEM_LOCK_REG_RANGE:
                            print("Mem tile lock init config")
                            dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
            if "R" in line:
                address=line.split(" addr=")[1].split(" ")[0].strip("\n")
                row=(int(address,0) & 0x1F00000) >> 20
                col=(int(address,0) & 0xFE000000) >> 25
                addr_offset=(int(address,0) & 0x000FFFFF)
                if row == SHIM_ROW:
                    IS_SHIM=True
                if row >= MEM_ROW_START and row <= MEM_ROW_END:
                    IS_MEM=True
                if row >= CORE_ROW_START :
                    IS_CORE=True
                if addr_offset in MEM_DMA_STATUS_REG_RANGE and addr_offset not in mem_mm2s_poll_dict.keys() and IS_MEM == True:
                            print("mem tile Poll DMA completion")
                            print("mem tile Poll channel")
                            print(hex(addr_offset))
                            mem_mm2s_poll_dict[addr_offset] = []
                            value=hex(int(next(file), 0) & 0x3F000000)
                            print("Poll value")
                            print(value)
                            mem_mm2s_poll_dict[addr_offset].append(value)
                            timeout="4294967295"
                            dump_frmt.write("# OPCODE : OP_READ32_POLL COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# REG_ADDR_OFFSET : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# REG_CMP_VAL  : "+str(value)+"\n")
                            dump_frmt.write("# LOOP_CNT  : "+str(timeout)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_READ32_POLL']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                            dump_frmt.write('{:08x}'.format(int(timeout,0))+"\n")
                elif addr_offset in SHIM_DMA_STATUS_REG_RANGE and (shim_tile_dma_poll == 1) and col in shim_tile_dma_poll_col and IS_SHIM == True:
                            print("Shim tile Poll DMA completion")
                            print("Shim tile Poll channel")
                            print(hex(addr_offset))
                            value=hex(int(next(file), 0) & 0x3F000000)
                            print("shim tile Poll value")
                            print(value)
                            timeout="4294967295"
                            dump_frmt.write("# OPCODE : OP_READ32_POLL COL : "+str(col)+" ROW : "+str(row)+"\n")
                            dump_frmt.write("# REG_ADDR_OFFSET : "+str(hex(addr_offset))+"\n")
                            dump_frmt.write("# REG_CMP_VAL  : "+str(value)+"\n")
                            dump_frmt.write("# LOOP_CNT  : "+str(timeout)+"\n")
                            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_READ32_POLL']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
                            dump_frmt.write('{:08x}'.format(int(timeout,0))+"\n")
                            shim_tile_dma_poll_col.remove(col)
                    #else:
                    #    dump_frmt.write("# OPCODE : OP_WRITE32 COL : "+str(col)+" ROW : "+str(row)+"\n")
                    #    dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
                    #    dump_frmt.write("# VALUE TO WRITE : "+str(value)+"\n")
                    #    dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_WRITE32']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
                    #    dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
                    #    dump_frmt.write('{:08x}'.format(int(value,0))+"\n")

            #if "R " in line:
            #    address=line.split(" addr=")[1].split(" ")[0].strip("\n")
            #    value=line.split(" data=")[1].split(" ")[0].strip("\n")
            #    row=(int(address,0) & 0x1F00000) >> 20
            #    col=(int(address,0) & 0xFE000000) >> 25
            #    addr_offset=(int(address,0) & 0x000FFFFF)
            #    if row == SHIM_ROW:
            #        IS_SHIM=True
            #    if row >= MEM_ROW_START and row <= MEM_ROW_END:
            #        IS_MEM=True
            #    if row >= CORE_ROW_START :
            #        IS_CORE=True
            #    if IS_SHIM == True:
            #        if addr_offset in SHIM_NOC_REG_RANGE1 or addr_offset in SHIM_NOC_REG_RANGE0 or addr_offset in SHIM_PL_REG_RANGE0 or addr_offset  in SHIM_PL_REG_RANGE1:                
            #            dump_frmt.write("# OPCODE : OP_READ32_CMP COL : "+str(col)+" ROW : "+str(row)+"\n")
            #            dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
            #            dump_frmt.write("# VALUE TO COMP : "+str(value)+"\n")
            #            dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_READ32_CMP']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00"+"\n")                        
            #            dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
            #            dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
            #    else:
            #        dump_frmt.write("# OPCODE : OP_READ32_CMP COL : "+str(col)+" ROW : "+str(row)+"\n")
            #        dump_frmt.write("# ADDRESS TO WRITE : "+str(hex(addr_offset))+"\n")
            #        dump_frmt.write("# VALUE TO COMP : "+str(value)+"\n")
            #        dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_READ32_CMP']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00"+"\n")                        
            #        dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
            #        dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
            #if "NOOP:" in line:
            #    address=line.split(":")[1].split(",")[0].strip("\n")                
            #    row=(int(address,0) & 0x00FF0000) >> 16
            #    col=(int(address,0) & 0x0000FF00) >> 8
            #    dump_frmt.write("# OPCODE : NOOP COL : "+str(col)+" ROW : "+str(row)+"\n")
            #    dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_NOOP']))+'{:02x}'.format(int(row))+'{:02x}'.format(int(col))+"00\n")

            #if "MP:" in line:
            #    address=line.split(":")[1].split(",")[0].strip("\n")
            #    value=line.split(":")[1].split(",")[2].strip("\n")
            #    row=(int(address,0) & 0x1F00000) >> 20
            #    col=(int(address,0) & 0xFE000000) >> 25
            #    addr_offset=(int(address,0) & 0x000FFFFF)
            #    timeout="4294967295"
            #    dump_frmt.write("# OPCODE : OP_READ32_POLL COL : "+str(col)+" ROW : "+str(row)+"\n")
            #    dump_frmt.write("# REG_ADDR_OFFSET : "+str(hex(addr_offset))+"\n")
            #    dump_frmt.write("# REG_CMP_VAL  : "+str(value)+"\n")
            #    dump_frmt.write("# LOOP_CNT  : "+str(value)+"\n")
            #    dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_READ32_POLL']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+"00\n")
            #    dump_frmt.write('{:08x}'.format(int(addr_offset))+"\n")
            #    dump_frmt.write('{:08x}'.format(int(value,0))+"\n")
            #    dump_frmt.write('{:08x}'.format(int(timeout,0))+"\n")
            #if "E:" in line:
            #    address=line.split(":")[1].split(",")[0].strip("\n")
            #    tct_value=line.split(":")[1].split(",")[1].strip("\n")
            #    print("address"+address)
            #    print("tct_Value"+'{:08x}'.format(int(tct_value,0)))
            #    col=(int(address,0) & 0x00FF0000) >> 19
            #    row=(int(address,0) & 0x0000FF00) >> 7
            #    type_dir=(int(address,0) & 0x000000FF)
            #    dump_frmt.write("# OPCODE : OP_SYNC COL : "+str(col)+" ROW : "+str(row)+"\n")
            #    dump_frmt.write('{:02x}'.format(int(dpu_opcodes['OP_SYNC']))+'{:02x}'.format(int(col))+'{:02x}'.format(int(row))+'{:02x}'.format(int(type_dir))+"\n")
            #    dump_frmt.write('{:08x}'.format(int(tct_value,0))+"\n")
            #    print(line)
    dump_frmt.close()


def ARRAY_IO_dump():  
    createDir(array_io_format_fldr)
    dump_frmt=open(array_io_format_fldr+"ARRAY_IO_"+tcl_file_name+".ini", "w")
    with open(axi_mm_dump_temp_fldr+"dump_"+tcl_file_name+".txt") as file:
        for line in file:
            if "W:" in line:
                if "MW:" in line:
                    address=line.split(":")[1].split(",")[0].strip("\n")
                    mask=line.split(":")[1].split(",")[1].strip("\n")
                    value=line.split(":")[1].split(",")[2].strip("\n")
                else:
                    address=line.split(":")[1].split(",")[0].strip("\n")
                    value=line.split(":")[1].split(",")[1].strip("\n")
                dump_frmt.write("WR32 "+address+":"+value+"\n")
            if "R:" in line:
                address=line.split(":")[1].split(",")[0].strip("\n")
                value=line.split(":")[1].split(",")[1].strip("\n")
                dump_frmt.write("RDC32 "+address+":"+value+"\n")
            if "MP:" in line:
                address=line.split(":")[1].split(",")[0].strip("\n")
                mask=line.split(":")[1].split(",")[1].strip("\n")
                value=line.split(":")[1].split(",")[2].strip("\n")
                dump_frmt.write("MASKPOLL "+address+":"+value+"\n")
    dump_frmt.close() 


def XSDB_dump():  
    createDir(xsdb_format_fldr)
    dump_frmt=open(xsdb_format_fldr+"XSDB_"+tcl_file_name+".ini", "w")
    with open(axi_mm_dump_temp_fldr+"dump_"+tcl_file_name+".txt") as file:
        for line in file:
            if "W:" in line:
                if "MW:" in line:
                    address=line.split(":")[1].split(",")[0].strip("\n")
                    value=line.split(":")[1].split(",")[2].strip("\n")
                else:
                    address=line.split(":")[1].split(",")[0].strip("\n")
                    value=line.split(":")[1].split(",")[1].strip("\n")
                dump_frmt.write("mwr -force "+address+" "+value+"\n")
            if "R:" in line:
                address=line.split(":")[1].split(",")[0].strip("\n")
                value=line.split(":")[1].split(",")[1].strip("\n")
                dump_frmt.write("mrd -force "+address+" "+value+"\n")
            if "MP:" in line:
                address=line.split(":")[1].split(",")[0].strip("\n")
                value=line.split(":")[1].split(",")[2].strip("\n")
                dump_frmt.write("mrd -force "+address+" "+value+"\n")
    dump_frmt.close() 


def AXI_MM_dump():  
    createDir(axi_mm_dump_fldr)
    dump_frmt=open(axi_mm_dump_fldr+"AXI-MM_dump_"+tcl_file_name+".txt", "w")
    with open(axi_mm_dump_temp_fldr+"dump_"+tcl_file_name+".txt") as file:
        for line in file:
            if "W:" in line:
                dump_frmt.write(line)
            if "R:" in line:
                print(line)
                dump_frmt.write(line)
            if "MP:" in line:
                print(line)
                dump_frmt.write(line)
            if "NOOP:" in line:
                dump_frmt.write(line)
            if "E:" in line:
                dump_frmt.write(line)
    dump_frmt.close() 

def execute_tcl():
    global tcl_file_path
    global axi_mm_dump_temp_fldr
    global axi_mm_dump_fldr
    global dpu_sequence_fldr
    global tcl_file_name
    tcl_file_name=tcl_file_path.rsplit('/',1)[1]
    os.system(X86_CLIEMB+" -i"+tcl_file_path+" -v | tee "+axi_mm_dump_temp_fldr+"dump_"+tcl_file_name+".txt > l.log")
    os.system("rm -rf *.log")



def set_vars(parser):
    global home_dir
    global RUN_MODE
    global FORMAT
    global X86_CLIEMB
    args = parser.parse_args()
    RUN_MODE = args.run_mode
    FORMAT = args.format
    TARGET = args.target
    if TARGET == "PHX" :
        X86_CLIEMB = X86_CLIEMB+"_AMD_PHX"
    if TARGET == "STX" :
        X86_CLIEMB = X86_CLIEMB+"_AMD_STRIX"
    os.environ["LD_LIBRARY_PATH"]=X86_CLIEMB_LIBS



def cleanDirs():
    global dpu_sequence_fldr
    os.system("rm -rf "+home_dir+"/*.csv")
    if os.path.exists(dpu_sequence_fldr+"dpu_sequence_RT_stack.ini"):
        os.remove(dpu_sequence_fldr+"dpu_sequence_RT_stack.ini")
#
#    if os.path.exists(xsdb_format_fldr):
#        shutil.rmtree(xsdb_format_fldr)
#
#    if os.path.exists(array_io_format_fldr):
#        shutil.rmtree(array_io_format_fldr)
#
#    if os.path.exists(amd_phoenix_fldr):
#        shutil.rmtree(amd_phoenix_fldr)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', choices=["BATCH","NORMAL", "CLEAN"],default="NORMAL", type=str, help='Options for running in the BATCH mode (i.e., running multiple tcl files) / NORMAL mode to run only single tcl file: Default is NORMAL mode')
    parser.add_argument('--format', choices=["ALL","AXI-MM", "XSDB","ARRAY_IO","DPU","PSS"],default="ALL", type=str, help='Output Format - ALL is for all formats(AXI-MM,XSDB,ARRAY_IO,DPU,PSS) \n AXI-MM for axi-mm reg read/write transactions \n XSDB for xsdb based sequence \n ARRAY_IO is for array_io format to run it in verif env \n DPU is for DPU sequence \n PSS is for PSS format to run it using pss flow')
    parser.add_argument('--target', choices=["PHX","STX"],default="STX", type=str, help='TARGET BOARD PHX for Phoenix, STX for Strix')
    #parser.add_argument('--tcl_file_path',required=True,type=str,help='specify tcl file path with tcl file if it is NORMAL Mode, if it is BATCH mode, just specify the path of the TCL file folders')  
    set_vars(parser)

    
    if RUN_MODE == "BATCH":
        if not os.path.isdir(axi_mm_dump_fldr):
            print(axi_mm_dump_fldr)
            print("AIE simulator output folder does not exist, run aiesim and generate AXI MM dump !!")
            sys.exit(1)

    if RUN_MODE == "NORMAL":
        if not os.path.isfile(tcl_file_path):
            print("Specify TCL testcase name path in the tcl_file_path option !!")
            sys.exit(1)

    if RUN_MODE == "CLEAN":
        cleanDirs()
        print("Workspace Cleaned")
    else:
        cleanDirs()
        #createDir(axi_mm_dump_temp_fldr)        
        #print("Generating AXI-MM Dump from TCL..")
        #execute_tcl()
        #AXI_MM_dump()
        #print("AXI-MM Dump Generated and output stored in : "+axi_mm_dump_fldr)
        #if FORMAT == "ALL" or FORMAT == "XSDB":
        #    print("Generating XSDB Format..")
        #    XSDB_dump()
        #    print("XSDB format Generated and output stored in : "+xsdb_format_fldr)
        #if FORMAT == "ALL" or FORMAT == "ARRAY_IO":
        #    print("Generating ARRAY_IO Format..")
        #    ARRAY_IO_dump()
        #    print("ARRAY IO format Generated and output stored in : "+array_io_format_fldr)
        if FORMAT == "ALL" or FORMAT == "DPU":
            print("Generating DPU Format..")
            dpu_sequence_dump()
            print("DPU format Generated and output stored in : "+dpu_sequence_fldr)
            #amd_phoenix_pss_dump_new()
            #print("PSS format Generated and output stored in : "+dpu_sequence_fldr)
        #if FORMAT == "ALL" or FORMAT == "PSS":
        #    print("Generating PSS Format..")
        #    #amd_phoenix_PSS_dump()
        #    amd_phoenix_pss_dump_new()
        #    print("PSS format Generated and output stored in : "+amd_phoenix_fldr)


if __name__ == "__main__":
    main()

