/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __AIEGRAPH_HPP__
#define __AIEGRAPH_HPP__

#include <fstream>
#include <ucode_gen.hpp>
#include <assert.h>
#include <nlohmann/json.hpp>
#include <utils/aie_control_parser.hpp>
#include <ps/op_buf.hpp>
#include <adf/adf_dbg.hpp>

using json = nlohmann::json;
using namespace std;

extern "C"{
    // custom function to workaround inability to clear transaction buffer in prepartion for next one; implementation exists in custom-built xaiengine.so 
    AieRC XAie_ClearTransaction(XAie_DevInst* DevInst);
};

namespace aiectrl
{

#define OPTIMIZE_TXN_BUFSIZE 1

class AIEGraphHandle
{
public:
    AIEGraphHandle() = delete;

    AIEGraphHandle(string json_filename, bool fw_dbg = false)
    {
        in_txn_ = false; 
        fw_dbg_ = fw_dbg;
        if (OPTIMIZE_TXN_BUFSIZE)
          fw_dbg_ = false;

        ifstream i(json_filename, ifstream::in);
        if (!i.good()) {
            fprintf(stderr, "Can't open file: %s\n",json_filename.c_str());
            exit(1);
        }

        json root;
        i >> root;

        if(root.empty()) {
            fprintf(stderr, "JSON Root is empty\n");
            exit(1);
        }

        auto driver_config = root["aie_metadata"]["driver_config"];
        auto mem_tile_row_start = driver_config.contains("mem_tile_row_start")?
          driver_config["mem_tile_row_start"].get<uint8_t>() : driver_config["reserved_row_start"].get<uint8_t>();
        auto mem_tile_num_rows = driver_config.contains("mem_tile_num_rows")?
          driver_config["mem_tile_num_rows"].get<uint8_t>() : driver_config["reserved_num_rows"].get<uint8_t>();

        int RC = XAIE_OK;
        XAie_Config config { 
                        driver_config["hw_gen"].get<uint8_t>(),               //xaie_dev_gen_aie
                        driver_config["base_address"].get<uint64_t>(),        //xaie_base_addr
                        driver_config["column_shift"].get<uint8_t>(),         //xaie_col_shift
                        driver_config["row_shift"].get<uint8_t>(),            //xaie_row_shift
                        driver_config["num_rows"].get<uint8_t>(),             //xaie_num_rows, 
                        driver_config["num_columns"].get<uint8_t>(),          //xaie_num_cols, 
                        driver_config["shim_row"].get<uint8_t>(),             //xaie_shim_row,
                        mem_tile_row_start,                                   //xaie_mem_tile_row_start, 
                        mem_tile_num_rows,                                    //xaie_mem_tile_num_rows,
                        driver_config["aie_tile_row_start"].get<uint8_t>(),   //xaie_aie_tile_row_start, 
                        driver_config["aie_tile_num_rows"].get<uint8_t>(),    //xaie_aie_tile_num_rows
                        {0}                                                   // PartProp
        };

        if (driver_config.contains("partition_overlay_start_cols") && driver_config.contains("partition_num_cols")) {
            std::vector<uint8_t> start_cols = driver_config["partition_overlay_start_cols"].get<std::vector<uint8_t>>();
            start_col_idx_ = *std::min_element(start_cols.begin(), start_cols.end());
            RC = XAie_SetupPartitionConfig(&devInst_val, 1 << driver_config["column_shift"].get<uint8_t>(), start_col_idx_, 
                                        driver_config["partition_num_cols"].get<uint8_t>());
            if(RC != XAIE_OK) {
                printf("Driver partition failed.\n");
                exit(1);
            }
        }


        RC = XAie_CfgInitialize(&devInst_val, &config);
        if(RC != XAIE_OK) {
            printf("Driver initialization failed.\n");
            exit(1);
        }

        auto gmios = root["aie_metadata"]["GMIOs"];
        const unsigned NUM_GMIO = gmios.size();

        if( NUM_GMIO <= 0) {
            fprintf(stderr, "No GMIOs found in the %s. NUM_GMIO=%d\n", json_filename.c_str(), NUM_GMIO);
            exit(1);
        }

        std::vector<GMIOConfig> gmio_cfg(NUM_GMIO);
        json_to_gmioconfig(&gmios, NUM_GMIO, gmio_cfg.data());
        shim_.resize(NUM_GMIO);

        if (fw_dbg_) std::cout << "FWDBG: Design has " << NUM_GMIO << " gmio ports" << std::endl;

        for (unsigned i = 0; i < NUM_GMIO; i++) {
            shim_[i] = make_shared<shimTileHandle>(&devInst_val, gmio_cfg[i] );

            if (fw_dbg_) std::cout << "FWDBG: Design has gmio port " <<  gmio_cfg[i].name << std::endl;

            gmio_name_map_[gmio_cfg[i].name] = i;
            if (gmio_cfg[i].type == gmio_config::gm2aie) gm2aie_.push_back(i);
            else if (gmio_cfg[i].type == gmio_config::aie2gm) aie2gm_.push_back(i);
        }

        auto dma_channels = root["aie_metadata"]["DMAChConfigs"];
        const unsigned NUM_MEMTILE_CHANNELS = dma_channels.size();

        if (NUM_MEMTILE_CHANNELS) {
            if (fw_dbg_) std::cout << "FWDBG: Design has " << NUM_MEMTILE_CHANNELS << " memtile ports" << std::endl;
            std::vector<DMAChannelConfig> dma_ch_cfg(NUM_MEMTILE_CHANNELS);
            json_to_dma_ch_config(&dma_channels, NUM_MEMTILE_CHANNELS, dma_ch_cfg.data());

            for (unsigned i = 0; i < NUM_MEMTILE_CHANNELS; i++) {
                if (fw_dbg_) std::cout << "FWDBG: Design has memtile port " << dma_ch_cfg[i].portName << std::endl;
                memtile_ports_[dma_ch_cfg[i].portName] = make_shared<memTilePortHandle>( &devInst_val, dma_ch_cfg[i]);
            }
        }
    }

    ~AIEGraphHandle()
    {
//        devInst_val = { 0 };
    }

    void enqueue_printop ( string str)
    {
        if (OPTIMIZE_TXN_BUFSIZE)
            return;

        ibuf_.addOP ( dbgPrint_op ( str ));
    }

    //TODO: Implement a new dump_transaction() to account for new txn format
    void dump_transaction ()
    {
        if (OPTIMIZE_TXN_BUFSIZE)
            return;
        /*
        XAie_TxnInst * txn = getTransaction(

        ibuf_.addOP(dbgPrint_op("Tid: " + std::to_string(txn->Tid) + 
                                " Flags: " + std::to_string(txn->Flags) + 
                                " NumCmd: " + std::to_string( txn->NumCmds) + 
                                " MaxCmds: " + std::to_string( txn->MaxCmds ) ));

        for (unsigned i = 0; i < txn->NumCmds; i++)
        {
            XAie_TxnCmd cmd = txn->CmdBuf[i];
            ibuf_.addOP(dbgPrint_op("Cmd: " + std::to_string(i) +
                " Op: " + std::to_string(cmd.Opcode) +
                " Mask: " + std::to_string(cmd.Mask) +
                " RegOff: " + std::to_string(cmd.RegOff) +                
                " Val: " + std::to_string(cmd.Value) +   
                " Ptr: " + std::to_string(cmd.DataPtr) +   
                " Sz: " + std::to_string(cmd.Size) ) );           
        }
        aiectrl::dump_transaction_aximm(txn, std::cout);
        */
    }
    err_code gm2aie_enqueuebd( unsigned idx, const void * addr, unsigned sz, uint32_t repeat_count = 1, bool enable_task_complete_token = false)
    {
        assert (idx < numGM2AIEPorts());
        startTransaction();
        err_code r = shim_[gm2aie_[idx]]->enqueue(addr, sz, repeat_count, enable_task_complete_token);
        if (r == err_code::ok){
            addTransaction();
        }
        return r;
    }

    err_code aie2gm_enqueuebd( unsigned idx, const void * addr, unsigned sz, uint32_t repeat_count = 1, bool enable_task_complete_token = false)
    {
        assert( idx < numAIE2GMPorts());
        startTransaction();
        err_code r = shim_[aie2gm_[idx]]->enqueue(addr,sz, repeat_count, enable_task_complete_token);
        if (r == err_code::ok) {
            addTransaction();
        }
        return r;
    }    
    
    err_code gm2aie_enqueuebd(unsigned idx, const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false, int32_t ext_buffer_idx=-1)
    {
        assert(idx < numGM2AIEPorts());
        return shimtile_enqueuebd ( shim_[gm2aie_[idx]], address, buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token, ext_buffer_idx );
    }

    err_code gm2aie_enqueuebd(const std::string & gmio_name, const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false, int32_t ext_buffer_idx=-1)
    {
        if (gmio_name_map_.find(gmio_name) == gmio_name_map_.end())
          throw std::invalid_argument("gm2aie_enqueuebd: Port " + gmio_name + " not found!");

        return shimtile_enqueuebd(getShimTilePort(gmio_name), address, buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token, ext_buffer_idx);
    }

    err_code aie2gm_enqueuebd(unsigned idx, const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false, int32_t ext_buffer_idx=-1)
    {
        assert(idx < numAIE2GMPorts());
        return shimtile_enqueuebd ( shim_[aie2gm_[idx]], address, buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token, ext_buffer_idx );
    }

    err_code aie2gm_enqueuebd(const std::string & gmio_name, const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false, int32_t ext_buffer_idx=-1)
    {
        if ( gmio_name_map_.find(gmio_name) == gmio_name_map_.end() )
            throw std::invalid_argument("aie2gm_enqueuebd: Port " + gmio_name + " not found!");

        return shimtile_enqueuebd( getShimTilePort(gmio_name) , address, buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token, ext_buffer_idx);
    }

    err_code memtile_enqueuebd(const std::string& dmaPortName, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false)
    {
        if (memtile_ports_.find(dmaPortName) == memtile_ports_.end()) 
            throw std::invalid_argument("memtile_enqueuebd: Port " + dmaPortName + " not found!");

        startTransaction();
        err_code r = memtile_ports_[dmaPortName]->enqueue_task(buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token);
        if (r == err_code::ok) {
            if (fw_dbg_)
            {
              ibuf_.addOP(dbgPrint_op("memtile_enqueuebd portName: " + dmaPortName));
              this->dump_transaction();
            }
            addTransaction();
        }
        return r;
    } 

    template<typename T>
    err_code gmio_wait_tct(T const & aie2gm_port) {
        if (fw_dbg_) ibuf_.addOP(dbgPrint_op("tct_wait"));
        startTransaction();

        std::vector<uint32_t> tct_config;
        if constexpr (std::is_same<T, unsigned>::value)
            tct_config = getAIE2GMPort(aie2gm_port)->wait_tct();
        else if constexpr (std::is_same<T, std::string>::value)
            tct_config = getShimTilePort(aie2gm_port)->wait_tct();
        else
            return errorMsg(err_code::aie_driver_error, "ERROR: Invalid type for gmio_wait_tct");

        if (tct_config.size() != 2) {
            return errorMsg(err_code::aie_driver_error, "ERROR: TCT config size is not 2");
        }

        tct_op_t tct_op;
        tct_op.word = tct_config[0];
        tct_op.config = tct_config[1];

        XAie_AddCustomTxnOp(&devInst_val, tct_op_code_, (void*)&tct_op, sizeof(tct_op));

        addTransaction();
        return err_code::ok;
    }

    void gmio_wait ( unsigned aie2gm_port_id)
    {
        if (fw_dbg_) ibuf_.addOP(dbgPrint_op("gmio_wait port id: " + std::to_string(aie2gm_port_id)));

        startTransaction();
        getAIE2GMPort(aie2gm_port_id)->wait();
        addTransaction();
    }

    void gmio_wait ( const std::string & port_name)
    {
        if (fw_dbg_) ibuf_.addOP(dbgPrint_op("gmio_wait port name: " + port_name));

        startTransaction();
        getShimTilePort(port_name)->wait();
        addTransaction();
    }

    err_code mem_wait_tct(const std::string & memtile_port) {
        if (fw_dbg_) ibuf_.addOP(dbgPrint_op("tct_wait"));
        startTransaction();

        std::vector<uint32_t> tct_config;
        tct_config = getMemTilePort(memtile_port)->wait_tct();

        if(tct_config.size() != 2) {
            throw std::invalid_argument("mem_wait_tct: TCT config size is not 2");
        }

        tct_op_t tct_op;
        tct_op.word = tct_config[0];
        tct_op.config = tct_config[1];

        XAie_AddCustomTxnOp(&devInst_val, tct_op_code_, (void*)&tct_op, sizeof(tct_op));

        addTransaction();
        return err_code::ok;
    }

    void mem_wait ( const std::string & port_name)
    {
        if (fw_dbg_) ibuf_.addOP(dbgPrint_op(" memtile_wait port name: " + port_name));
        startTransaction();
        getMemTilePort(port_name)->wait();
        addTransaction();
    }

    err_code lock_set_value(uint32_t column, uint32_t row, uint8_t lock_id, int8_t lock_value)
    {
        startTransaction();
        AieRC RC = XAie_LockSetValue(&devInst_val, XAie_TileLoc(column, row), XAie_LockInit(lock_id, lock_value));
        if (RC == AieRC::XAIE_OK) {
            if (fw_dbg_)
            {
              ibuf_.addOP(
                dbgPrint_op("lock_set_value: (" + std::to_string(column) + ", " + std::to_string(row) + ") - lock_id: " 
                            + std::to_string(lock_id) + " - lock_value: " + std::to_string(lock_value))
              );
              this->dump_transaction();
            }
        } else {
            return errorMsg(err_code::aie_driver_error, "ERROR: LockSetValue: AIE driver error.");
        }
        addTransaction();
        return err_code::ok;
    }

    const op_buf& getInstrBuffer() {
        if (OPTIMIZE_TXN_BUFSIZE && in_txn_ == true) {
            addTransaction(true);
        }

        return ibuf_;
    }

    std::shared_ptr<shimTileHandle> getGM2AIEPort(unsigned idx) const { 
        assert (idx < numGM2AIEPorts());
        return shim_[gm2aie_[idx]]; 
    }
    std::shared_ptr<shimTileHandle> getAIE2GMPort(unsigned idx) const { 
        assert (idx < numAIE2GMPorts());
        return shim_[aie2gm_[idx]]; 
    }

    std::shared_ptr<shimTileHandle> getShimTilePort ( const std::string & gmio_name ) const
    {
        if (gmio_name_map_.find(gmio_name) == gmio_name_map_.end())
            throw std::invalid_argument("getShimTilePort: Port " + gmio_name + " not found!");

        return shim_.at(gmio_name_map_.at(gmio_name));
    }

    std::shared_ptr<memTilePortHandle> getMemTilePort( const std::string & name) const
    {
        if (memtile_ports_.find(name) == memtile_ports_.end())
            throw std::invalid_argument("getMemTilePort: Port " + name + " not found!");

        return memtile_ports_.at(name);
    }

    unsigned numGM2AIEPorts() const{ return gm2aie_.size(); }

    unsigned numAIE2GMPorts() const{ return aie2gm_.size(); }

    unsigned numMemTilePorts() const { return memtile_ports_.size(); }

    void clearTransactionBuffer() { 
      ibuf_ = op_buf(); 
      assert(in_txn_ == false);

      for (unsigned i=0; i < shim_.size(); i++)
        shim_[i]->clearJournal();

      for (auto i : memtile_ports_)
        i.second->clearJournal();
    }

    const uint32_t getStartColIdx() const { return start_col_idx_; }

private:
    void addTransaction(bool force_txn = false)
    {
        if (!OPTIMIZE_TXN_BUFSIZE || force_txn) {
            ibuf_.addOP(transaction_op(getTransaction(), fw_dbg_));
        }
    }

    void GenerateCustomOpCodes()
    {
        // Generate all custom opcodes here. 
        tct_op_code_ = XAie_RequestCustomTxnOp(&devInst_val);
        patch_op_code_ = XAie_RequestCustomTxnOp(&devInst_val);
        read_registers_op_code_ = XAie_RequestCustomTxnOp(&devInst_val);
    }

    void startTransaction() 
    {
        if (in_txn_ == false) {
            XAie_StartTransaction(&devInst_val, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
            GenerateCustomOpCodes();
            in_txn_ = true;
        }
    }

    err_code shimtile_enqueuebd(std::shared_ptr<shimTileHandle> tileHandle, const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false, int32_t ext_buffer_idx=-1)
    {
        startTransaction();
        err_code r = tileHandle->program_only(address, buffer_descriptors, bd_ids, repeat_count, enable_task_complete_token);
        if (r == err_code::ok && ext_buffer_idx != -1) {
          auto[regAddr, r] = tileHandle->compute_patch_regaddr(bd_ids[0]);
          if (r == err_code::ok) {
            patch_op_t op;
            op.action = 0; // patch shim
            op.regaddr = regAddr;
            op.argidx = ext_buffer_idx;

            // if patch mode, the 'address' provided is an offset
            // this will be added with a real addr at runtime
            op.argplus = reinterpret_cast<u64>(address); 

            XAie_AddCustomTxnOp(&devInst_val, patch_op_code_, 
              (void*)&op, sizeof(op));
          }
        }
        if (r == err_code::ok) {
          r = tileHandle->enqueuetask_only(bd_ids[0], repeat_count, enable_task_complete_token);
        }
        if (r == err_code::ok) {
            if ( fw_dbg_ ) this->dump_transaction();
            addTransaction();
        } else{
          throw std::invalid_argument ( "shimtile_enqueuebd transaction logging failure");
        }
        return r;
    }

    void* getTransaction(bool clear_txn = true) { 
        void* txn = XAie_ExportSerializedTransaction(&devInst_val, 1, 0); 
        if(clear_txn) {
            XAie_ClearTransaction(&devInst_val);
            in_txn_ = false; // clear
        }
        return txn;
    }

    XAie_DevInst devInst_val = { 0 };

    std::vector<unsigned> gm2aie_, aie2gm_;
    std::vector< std::shared_ptr<shimTileHandle> > shim_;    
    std::unordered_map<std::string, unsigned> gmio_name_map_;
    std::unordered_map<std::string, std::shared_ptr<memTilePortHandle> > memtile_ports_;
    op_buf ibuf_;
    bool fw_dbg_;
    bool in_txn_;
    int32_t tct_op_code_;
    int32_t patch_op_code_;
    int32_t read_registers_op_code_;
    // This field denotes that starting column for a partition
    uint32_t start_col_idx_{0};
};

}


#endif
