//! \file
//! \brief Initialisation for bio spiking symmetric 
#include "bio_spiking_symmetric_stdp.h"

//---------------------------------------
// Globals
//---------------------------------------
// Exponential lookup-tables
int16_lut *tau_plus_lookup;

//---------------------------------------
// Functions
//---------------------------------------
address_t timing_initialise(address_t address) {
    // **TODO** assert number of neurons is less than max

    // Copy LUTs from following memory
    address_t lut_address = address;
    tau_plus_lookup = maths_copy_int16_lut(&lut_address);
    tau_minus_lookup = maths_copy_int16_lut(&lut_address);
    tau_x_lookup = maths_copy_int16_lut(&lut_address);
    tau_y_lookup = maths_copy_int16_lut(&lut_address);

    return lut_address;
}

