#ifndef _BIO_SPIKING_SYMMETRIC_STDP_H_
#define _BIO_SPIKING_SYMMETRIC_STDP_H_

#include <neuron/plasticity/stdp/synapse_structure/synapse_structure_weight_impl.h>
#include "timing.h"
#include <neuron/plasticity/stdp/weight_dependence/weight_two_term.h>
#include <debug.h>
#include <neuron/plasticity/stdp/maths.h>
#include <neuron/plasticity/stdp/stdp_typedefs.h>

// External Variables
extern int16_lut *tau_symm_lookup;

/**
Creates a struct to store a symmetric trace
 */
typedef struct symm_trace_t {
    int32_t trace;
    uint32_t last_spike_time;
} symm_trace_t;


static inline symm_trace_t initialize_symm_trace(void){
    return (symm_trace_t){
        .trace = 0
    };
}

/**
Updates the post-synaptic trace: performs exponential decay and increments the trace in case of spike
 */
static inline symm_trace_t timing_decay_post(uint32_t current_time, uint32_t last_time, symm_trace_t last_trace){
    uint32_t dt = current_time - last_time;
    int32_t decayed_trace = STDP_FIXED_MUL_16X16(last_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    return (symm_trace_t) {.trace = decayed_trace, .last_spike_time = last_trace.last_spike_time};
}


static inline symm_trace_t timing_decay_pre(uint32_t current_time, uint32_t last_time, symm_trace_t last_trace){
    uint32_t dt = current_time - last_time;
    int32_t decayed_trace = STDP_FIXED_MUL_16X16(last_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    return (symm_trace_t) {.trace = decayed_trace, .last_spike_time = last_trace.last_spike_time};
}

static inline symm_trace_t timing_add_post_spike(uint32_t time, uint32_t last_time, symm_trace_t last_trace, uint16_t invtau_symm){
    symm_trace_t temp = timing_decay_post(time, last_time, last_trace);
    temp.trace = temp.trace + invtau_symm;
    temp.last_spike_time = time;
    return temp;
}

static inline symm_trace_t timing_add_pre_spike(uint32_t time, uint32_t last_time, symm_trace_t last_trace, uint16_t invtau_symm){
    symm_trace_t temp = timing_decay_pre(time, last_time, last_trace);
    temp.trace = temp.trace + invtau_symm;
    temp.last_spike_time = time;
    return temp;
}

static inline update_state_t timing_apply_pre_spike(uint32_t time, UNUSED uint32_t last_pre_time, UNUSED symm_trace_t last_pre_trace, uint32_t last_post_time, symm_trace_t last_post_trace, update_state_t previous_state, double symm_amp, double learning_rate){
    uint32_t dt = time - last_post_time;
    int32_t decayed_post = STDP_FIXED_MUL_16X16(last_post_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    double gain = (double)(symm_amp * learning_rate);
    double val = gain*(double)decayed_post;
    return weight_one_term_apply_potentiation(previous_state, (int32_t) val);  
}

static inline update_state_t timing_apply_post_spike(uint32_t time, uint32_t last_pre_time, symm_trace_t last_pre_trace, UNUSED uint32_t last_post_time, UNUSED symm_trace_t last_post_trace, update_state_t previous_state, double symm_amp, double learning_rate){
    uint32_t dt = time - last_pre_time;
    int32_t decayed_pre = STDP_FIXED_MUL_16X16(last_pre_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    double gain = (double)(symm_amp * learning_rate);
    double val = gain*(double)decayed_pre;
    return weight_one_term_apply_potentiation(previous_state, (int32_t)val); 
}

