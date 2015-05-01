// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "scoreboard.h"
#include "shader.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader_trace.h"


//Constructor
Scoreboard::Scoreboard( unsigned sid, unsigned n_warps )
: longopregs()
{
	m_sid = sid;
	//Initialize size of table
	reg_table.resize(n_warps);
	longopregs.resize(n_warps);
}

// Print scoreboard contents
void Scoreboard::printContents() const
{
	printf("scoreboard contents (sid=%d): \n", m_sid);
	for(unsigned i=0; i<reg_table.size(); i++) {
		if(reg_table[i].size() == 0 ) continue;
		printf("  wid = %2d: ", i);
		std::deque<reg_and_mask>::const_iterator it;
		for( it=reg_table[i].begin() ; it != reg_table[i].end(); it++ ){
			printf("%u (", it->reg);
			for (unsigned j=0; j<it->active_mask.size(); j++){
				printf("%c", (it->active_mask.test(j)?'1':'0') );
			}
			printf(") ");
			
		}
		printf("\n");
	}
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum, const active_mask_t & active_mask) 
{
	unsigned reserved = 0;
	for (std::deque<reg_and_mask>::iterator it=reg_table[wid].begin(); it!=reg_table[wid].end(); ++it)
	{
		if (it->reg == regnum){
			if ((it->active_mask & active_mask).any()){
				printf("Error: trying to reserve an already reserved register (sid=%d, wid=%d, regnum=%d).", m_sid, wid, regnum);
				abort();
			}
			else{
				SHADER_DPRINTF( SCOREBOARD,
                    "Updating mask - warp:%d, reg: %d\n", wid, regnum );
				
				it->active_mask |= active_mask;

				reserved = 1;
			}
		}
	}
	
    SHADER_DPRINTF( SCOREBOARD,
                    "Reserved Register - warp:%d, reg: %d\n", wid, regnum );
	reg_and_mask new_entry;
	new_entry.reg = regnum;
	new_entry.active_mask = active_mask;
	//reserve new entry
	if (reserved == 0){
		reg_table[wid].push_back(new_entry);
	}
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum, const active_mask_t & active_mask) 
{
	if (reg_table[wid].size() == 0)
		return;
	
	for (std::deque<reg_and_mask>::iterator it=reg_table[wid].begin(); it!=reg_table[wid].end(); ++it)
	{
		if (it->reg == regnum){
			
			//Sanity check
			for (unsigned i = 0; i < active_mask.size(); i++){
				//if we're resetting a lane that was never set
				if (active_mask[i] && !it->active_mask[i]){
					printf("Error: trying to reset lanes that were never reserved (sid=%d, wid=%d, regnum=%d).\n", m_sid, wid, regnum);
					abort();		
				}
			}
			
			SHADER_DPRINTF( SCOREBOARD,
                    "Release register - warp:%d, reg: %d\n", wid, regnum );
			
			//otherwise remove the mask	
			it->active_mask &= ~active_mask;
			//remove entry entirely if nothing is left
			if (it->active_mask.none()){
				reg_table[wid].erase(it);
			}
			
			break;
		}
	}
	
	//baseline was that if we didn't find it we just returned, so we respect it here as well

}

const bool Scoreboard::islongop (unsigned warp_id,unsigned regnum) {
	return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst) 
{
    for( unsigned r=0; r < 4; r++) {
        if(inst->out[r] > 0) {
            reserveRegister(inst->warp_id(), inst->out[r], inst->get_active_mask());
            SHADER_DPRINTF( SCOREBOARD,
                            "Reserved register - warp:%d, reg: %d\n",
                            inst->warp_id(),
                            inst->out[r] );
        }
    }

    //Keep track of long operations
    if (inst->is_load() &&
    		(	inst->space.get_type() == global_space ||
    			inst->space.get_type() == local_space ||
                inst->space.get_type() == param_space_kernel ||
                inst->space.get_type() == param_space_local ||
                inst->space.get_type() == param_space_unclassified ||
    			inst->space.get_type() == tex_space)){
    	for ( unsigned r=0; r<4; r++) {
    		if(inst->out[r] > 0) {
                SHADER_DPRINTF( SCOREBOARD,
                                "New longopreg marked - warp:%d, reg: %d\n",
                                inst->warp_id(),
                                inst->out[r] );
                longopregs[inst->warp_id()].insert(inst->out[r]);
            }
    	}
    }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t *inst) 
{
    for( unsigned r=0; r < 4; r++) {
        if(inst->out[r] > 0) {
            SHADER_DPRINTF( SCOREBOARD,
                            "Register Released - warp:%d, reg: %d\n",
                            inst->warp_id(),
                            inst->out[r] );
            releaseRegister(inst->warp_id(), inst->out[r], inst->get_active_mask());
            longopregs[inst->warp_id()].erase(inst->out[r]);
        }
    }
}

/** 
 * Checks to see if registers used by an instruction are reserved in the scoreboard
 *  
 * @return 
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/ 
bool Scoreboard::checkCollision( unsigned wid, const class inst_t *inst, const active_mask_t &active_mask ) const
{
	// Get list of all input and output registers
	std::set<unsigned> inst_regs;

	if(inst->out[0] > 0) inst_regs.insert(inst->out[0]);
	if(inst->out[1] > 0) inst_regs.insert(inst->out[1]);
	if(inst->out[2] > 0) inst_regs.insert(inst->out[2]);
	if(inst->out[3] > 0) inst_regs.insert(inst->out[3]);
	if(inst->in[0] > 0) inst_regs.insert(inst->in[0]);
	if(inst->in[1] > 0) inst_regs.insert(inst->in[1]);
	if(inst->in[2] > 0) inst_regs.insert(inst->in[2]);
	if(inst->in[3] > 0) inst_regs.insert(inst->in[3]);
	if(inst->pred > 0) inst_regs.insert(inst->pred);
	if(inst->ar1 > 0) inst_regs.insert(inst->ar1);
	if(inst->ar2 > 0) inst_regs.insert(inst->ar2);

	// Check for collision, get the intersection of reserved registers and instruction registers
	std::set<unsigned>::const_iterator it2;
	for ( it2=inst_regs.begin() ; it2 != inst_regs.end(); it2++ ){
		for (std::deque<reg_and_mask>::const_iterator it=reg_table[wid].begin(); it!=reg_table[wid].end(); ++it)
		{
			/*if (it->reg == *it2){
				return true;
			}*/
			//only collides when masks intersect
			if (it->reg == *it2){
				if ((it->active_mask & active_mask).any()){
					return true;
				}
				//otherwise have to check other registers
			}
		}
	}
	//none of the registers triggering 
	return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const
{
	return !reg_table[wid].empty();
}
