//! \file
//! \brief Initialisation for timing_pfister_triplet_impl.h
#include "bio_spiking_triplet_stdp.h"

//---------------------------------------
// Globals
//---------------------------------------
// Exponential lookup-tables
int16_lut *tau_plus_lookup;
int16_lut *tau_minus_lookup;
int16_lut *tau_x_lookup;
int16_lut *tau_y_lookup;
double *pot_amp;
double *learning_rate;
double *dep_amp;
uint16_t *invtau_slow;
uint16_t *invtau_fast;

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
