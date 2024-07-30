/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

/*
 * Copyright 2022 Xilinx, Inc
 *
 */


#ifdef __AIESIM_TCT__

#ifndef __DPUFW_AIE_TCT_AIESIM_H__
#define __DPUFW_AIE_TCT_AIESIM_H__

//#include <iostream>
//#include <iomanip>
//#include <fstream>

#include <adf.h>

#define AIE_TCT_INVALID         0
#define AIE_TCT_VALID           1
#define AIE_TCT_RCVD            2

#define AIE_TCT_PKT_TYPE        0x6

#define AIE_TCT_ACTORID_SHIFT   8U
#define AIE_TCT_PKT_TYPE_SHIFT  12U
#define AIE_TCT_SRC_ROW_SHIFT   16U
#define AIE_TCT_SRC_COL_SHIFT   21U

#define AIE_TCT_ACTORID_MASK	(0xF << AIE_TCT_ACTORID_SHIFT)
#define AIE_TCT_PKT_TYPE_MASK	(0x7 << AIE_TCT_PKT_TYPE_SHIFT)
#define AIE_TCT_SRC_ROW_MASK	(0x1F << AIE_TCT_SRC_ROW_SHIFT)
#define AIE_TCT_SRC_COL_MASK	(0x7F << AIE_TCT_SRC_COL_SHIFT)

#define AIE_TCT_MASK (AIE_TCT_ACTORID_MASK | AIE_TCT_PKT_TYPE_MASK | \
		AIE_TCT_SRC_COL_MASK | AIE_TCT_SRC_ROW_MASK)

#ifndef XAIE_NUM_ROWS
#define XAIE_NUM_ROWS           6U
#endif

#ifndef AIE_TCT_MAX_ACTORID
#define AIE_TCT_MAX_ACTORID     12
#endif

namespace aiesim_tct {

struct aie_tct
{
    uint8_t valid;
    uint32_t tct;
};

struct aie_tct aie_tct_create(uint8_t actor_id, uint8_t src_row, uint8_t src_col)
{
    struct aie_tct tct;

    tct.tct = (actor_id << AIE_TCT_ACTORID_SHIFT) |
              (src_row << AIE_TCT_SRC_ROW_SHIFT)  |
              (src_col << AIE_TCT_SRC_COL_SHIFT)  |
              (AIE_TCT_PKT_TYPE << AIE_TCT_PKT_TYPE_SHIFT);
    tct.tct &= AIE_TCT_MASK;
    tct.valid = AIE_TCT_VALID;

    return tct;
}

void aie_tct_push_buff_to_map(uint32_t fifo_id, uint8_t *map)
{
    // Maintain last file position for each fifo to avoid re-reading already processed TCTs.
    static std::unordered_map<uint32_t, std::streampos> LAST_POS_MAP;
    auto last_pos = LAST_POS_MAP.find(fifo_id);
    if (last_pos == LAST_POS_MAP.end())
    {
        bool success = false;
        std::tie(last_pos, success) = LAST_POS_MAP.emplace(fifo_id, 0);
        assert(success);
    }

    std::string filename = "./aiesimulator_output/tct" + std::to_string(fifo_id) + ".txt";
    std::ifstream tct_file(filename);
    if (tct_file)
    {
        tct_file.seekg(last_pos->second, std::ios_base::beg);

        std::string line;
        while (std::getline(tct_file, line))
        {
            last_pos->second = tct_file.tellg();

            if (line.empty() || line[0] == 'T')
            {
                continue;
            }

            try
            {
                uint32_t token = std::stoul(line);

                std::cout << "TCT received: 0x" << std::hex << token << std::endl;

                uint8_t row, col, actor_id, pkt_type;

                row = (token & AIE_TCT_SRC_ROW_MASK) >>
                    AIE_TCT_SRC_ROW_SHIFT;
                col = (token & AIE_TCT_SRC_COL_MASK) >>
                    AIE_TCT_SRC_COL_SHIFT;
                actor_id = (token & AIE_TCT_ACTORID_MASK) >>
                    AIE_TCT_ACTORID_SHIFT;
                pkt_type = (token & AIE_TCT_PKT_TYPE_MASK) >>
                    AIE_TCT_PKT_TYPE_SHIFT;
                if ((pkt_type == AIE_TCT_PKT_TYPE) && (token != 0x77777777))
                {
                    ++map[(col * XAIE_NUM_ROWS + row) * AIE_TCT_MAX_ACTORID + actor_id];
                }
            }
            catch (const std::invalid_argument &)
            {
                std::cout << "Not a valid TCT: " << line << std::endl;
            }
        }
    }
    else
    {
        std::cout << "Cannot open TCT fifo stream: " << filename << std::endl;
    }
}

void aie_tct_if_in_map(uint8_t *map, struct aie_tct *expected_tcts, uint32_t num_tct, uint32_t *pending_tct)
{
    for (uint32_t i = 0; i < num_tct; i++)
    {
        uint8_t row, col, actor_id;

        row = (expected_tcts[i].tct & AIE_TCT_SRC_ROW_MASK) >>
            AIE_TCT_SRC_ROW_SHIFT;
        col = (expected_tcts[i].tct & AIE_TCT_SRC_COL_MASK) >>
            AIE_TCT_SRC_COL_SHIFT;
        actor_id = (expected_tcts[i].tct & AIE_TCT_ACTORID_MASK) >>
            AIE_TCT_ACTORID_SHIFT;
        if (map[(col * XAIE_NUM_ROWS + row) * AIE_TCT_MAX_ACTORID + actor_id] > 0U)
        {
            expected_tcts[i].valid = AIE_TCT_RCVD;
            --map[(col * XAIE_NUM_ROWS + row) * AIE_TCT_MAX_ACTORID + actor_id];
            --(*pending_tct);
        }
        if (*pending_tct == 0)
        {
            return;
        }
    }
}

int aie_tct_map_wait(uint8_t *map, uint32_t fifo_id, struct aie_tct *expected_tcts, uint32_t num_tct)
{
    uint32_t pending_tct = num_tct;

    aie_tct_if_in_map(map, expected_tcts, num_tct, &pending_tct);
    while (pending_tct > 0)
    {
        // Spin the simulation to give a chance to produce new TCTs.
        wait(20, SC_NS);

        aie_tct_push_buff_to_map(fifo_id, map);
        aie_tct_if_in_map(map, expected_tcts, num_tct, &pending_tct);
    }

    return 0;
}

} // aiesim_tct

#endif //__DPUFW_AIE_TCT_AIESIM_H__

#endif //__AIESIM_TCT__
